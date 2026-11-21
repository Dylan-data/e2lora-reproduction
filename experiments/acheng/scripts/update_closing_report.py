#!/usr/bin/env python3
"""Update the member-three report from audited local experiment artifacts."""

import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

from lxml import etree


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments" / "acheng"
REPORT = TASK / "E²LoRA 论文复现报告.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def text(element):
    return "".join(element.xpath(".//w:t/text()", namespaces=NS)).strip()


def target_table(body, headers):
    matches = [
        table
        for table in body.xpath("./w:tbl", namespaces=NS)
        if (rows := table.xpath("./w:tr", namespaces=NS))
        and [text(cell) for cell in rows[0].xpath("./w:tc", namespaces=NS)] == headers
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one table with headers {headers}, got {len(matches)}")
    return matches[0]


def paragraph_after(body, prefix, content):
    matches = [
        paragraph for paragraph in body.xpath("./w:p", namespaces=NS)
        if text(paragraph).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one paragraph starting {prefix!r}, got {len(matches)}")
    insert_after(body, matches[0], content)


def insert_after(body, anchor, content):
    paragraph = etree.Element(f"{{{W}}}p")
    run = etree.SubElement(paragraph, f"{{{W}}}r")
    node = etree.SubElement(run, f"{{{W}}}t")
    node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = content
    body.insert(body.index(anchor) + 1, paragraph)


def main():
    paths = {
        "eurosat_zero": "experiments/acheng/outputs/EuroSAT/zero-shot/2026/54495-audited/result.json",
        "eurosat_lora": "experiments/acheng/outputs/EuroSAT/LoRA-r8/2026/20261117T140537+0800/result.json",
        "dtd_zero": "experiments/acheng/outputs/DTD/zero-shot/2026/20261119T162826+0800/result.json",
        "dtd_lora": "experiments/acheng/outputs/DTD/LoRA-r8/2026/20261119T173117+0800/result.json",
        "gtsrb_zero": "experiments/acheng/outputs/GTSRB/zero-shot/2026/20261119T191610+0800/result.json",
        "gtsrb_lora": "experiments/acheng/outputs/GTSRB/LoRA-r8/2026/20261119T194115+0800/result.json",
    }
    results = {name: read_json(path) for name, path in paths.items()}
    expected = {
        "eurosat_zero": (1424, 2700), "eurosat_lora": (2673, 2700),
        "dtd_zero": (805, 1880), "dtd_lora": (1501, 1880),
        "gtsrb_zero": (4730, 12630), "gtsrb_lora": (12441, 12630),
    }
    for name, (correct, count) in expected.items():
        result = results[name]
        test = result if name == "eurosat_zero" else result["test"]
        accuracy = test.get("accuracy", test.get("accuracy_percent"))
        if (test["correct"], test["count"]) != (correct, count) or abs(accuracy - 100 * correct / count) > 1e-8:
            raise ValueError(f"Unexpected score in {paths[name]}")
        if len(test["predictions"]) != count or len(test["labels"]) != count:
            raise ValueError(f"Incomplete predictions in {paths[name]}")
        if sum(a == b for a, b in zip(test["predictions"], test["labels"])) != correct:
            raise ValueError(f"Prediction mismatch in {paths[name]}")
    for name in ("eurosat_lora", "dtd_lora", "gtsrb_lora"):
        result = results[name]
        if (result["seed"], result["rank"], result["alpha"], result["adapter_trainable_parameters"],
            result["total_trainable_parameters"], result["max_epochs"]) != (2026, 8, 16, 1327104, 1327104, 30):
            raise ValueError(f"Unexpected LoRA settings in {paths[name]}")
        if not result["frozen_parameters_unchanged"]:
            raise ValueError(f"Frozen parameter verification failed in {paths[name]}")
    audit = read_json("experiments/acheng/reports/core-audit.json")["e2lora_blocker"]
    if (audit["rank_sum_total"], audit["recomputed_unique_shared_adapter_parameters"]) != (575, 1325568):
        raise ValueError("Unexpected E²LoRA audit data")

    backup = TASK / "reports" / f"E²LoRA 论文复现报告-收尾前备份-{datetime.now().strftime('%Y%m%dT%H%M%S%z')}.docx"
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(REPORT, backup)
    original_hash = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if hashlib.sha256(backup.read_bytes()).hexdigest() != original_hash:
        raise ValueError("Backup checksum mismatch")

    with ZipFile(REPORT) as archive:
        parts = {info.filename: (info, archive.read(info.filename)) for info in archive.infolist()}
    root = etree.fromstring(parts["word/document.xml"][1])
    original = copy.deepcopy(root)
    body = root.find("w:body", NS)
    if any("成员三收尾：" in text(p) for p in body.xpath("./w:p", namespaces=NS)):
        raise ValueError("Closing annotations already exist; refuse to duplicate them")

    table3 = target_table(body, ["方法", "参数", "Cars", "DTD", "EuroSAT", "GTSRB", "RESISC45", "SUN397", "SVHN", "均值"])
    rows = {text(cells[0]): [text(c) for c in cells] for row in table3.xpath("./w:tr", namespaces=NS)
            if (cells := row.xpath("./w:tc", namespaces=NS))}
    assert [rows["Zero-shot"][i] for i in (3, 4, 5)] == ["44.39／42.82", "42.22／52.74", "35.22／37.45"]
    assert [rows["LoRA"][i] for i in (1, 3, 4, 5)] == ["1.31M／1.33M", "76.88／79.84", "98.42／99.00", "97.10／98.50"]

    table13 = target_table(body, ["方法与基础秩", "图像参数", "图像准确率", "文本参数", "文本均分"])
    lora_rows = [row for row in table13.xpath("./w:tr", namespaces=NS)
                 if text(row.xpath("./w:tc", namespaces=NS)[0]) == "LoRA 秩 8"]
    if len(lora_rows) != 1:
        raise ValueError("Cannot uniquely locate table 13 LoRA rank-8 row")
    cell = lora_rows[0].xpath("./w:tc", namespaces=NS)[1]
    nodes = cell.xpath(".//w:t", namespaces=NS)
    if len(nodes) != 1 or nodes[0].text != "1.31M／____":
        raise ValueError("Unexpected table 13 image-parameter cell")
    nodes[0].text = "1.31M／1.33M"

    image_heading = next(p for p in body.xpath("./w:p", namespaces=NS) if text(p) == "4  主结果  图像分类")
    candidates = [p for p in body.xpath("./w:p", namespaces=NS)
                  if body.index(p) > body.index(image_heading) and text(p).startswith("本组结论与设置差异")]
    if not candidates:
        raise ValueError("Image conclusion anchor not found")
    image_anchor = candidates[0]
    details = [
        "成员三收尾：三组 LoRA 相对 Zero-shot 分别提高 EuroSAT 46.26、DTD 37.02、GTSRB 61.05 个百分点（按表中两位小数计算）。这些是固定镜像划分上的单种子结果，不代表跨种子稳定性；镜像与论文原始划分的等价性未建立，不能据此作严格论文优劣结论。七数据集不齐，不计算七项均值。",
        "成员三收尾：模型 openai/clip-vit-base-patch16，原始 revision 57c216476eefef5ab752ec549e440a49ae4ae5f3；核验一致的 safetensors revision 5ef227a78de3f75873f373246dac80def63b0003。普通 LoRA 更新视觉 Transformer 的 12 层各自 q/k/v/out（768×768）、fc1（3072×768）、fc2（768×3072），共 72 个线性层；不含 visual_projection。各层独立适配、不共享，物理秩及可更新秩均为 8，alpha=16，无适配器 bias；冻结基础模型和文本分支，冻结参数核验通过。适配参数及总可训练参数均 1,327,104。此设置仅适用于已运行的 LoRA，不能作为 E²LoRA 已确定的设置。",
        "成员三收尾：采用 FP32、Adam、学习率 1e-4、betas=(0.9,0.999)、有效 batch=64、weight decay=0、warmup=0.3；30 epochs 上限、先线性预热再恒定学习率、无随机增强均为本组选择。训练/评估 seed 均为 2026。提示词为 a photo of a {class}；类别顺序按固定镜像标签核对。CLIP 处理器 RGB、短边缩放 224、中心裁剪 224×224，rescale 1/255，mean=(0.48145466,0.4578275,0.40821073)，std=(0.26862954,0.26130258,0.27577711)。",
        "成员三收尾：三个镜像及 revision：EuroSAT tanganke/eurosat@43750fc422c7fbd4289e2df3f7473617a5937acc，10 类；DTD tanganke/dtd@d2afa97d9f335b1a6b3b09c637aef667f98f966e，47 类；GTSRB tanganke/gtsrb@ea889fff70dda7b09e94a4b207dded438deca549，43 类。分别从镜像 train 按类分层留出 10% 验证，训练/验证/测试样本数依次为 19,440/2,160/2,700、3,384/376/1,880、23,976/2,664/12,630；测试集不参与调参。按验证准确率选择最佳 checkpoint，三者最佳 epoch 分别为 25、20、26，随后各在完整测试集评分。",
        "成员三收尾：LoRA 原始结果记录的训练/最终评估秒数和总 GPU 小时依次为 EuroSAT 6449.70/416.46/2.331（A40，峰值 11,801,475,072 bytes）、DTD 1722.94/157.52/0.538（A40，峰值 11,801,550,848 bytes）、GTSRB 7699.19/517.65/2.337（结果记录 GPU 为 L40S，峰值 11,801,542,656 bytes）；总时间还包含准备等开销。分段运行的 GPU 型号不应仅据最终记录推广到所有分段。零样本：EuroSAT A40，记录 elapsed 20.99 秒、峰值 771,623,424 bytes；DTD A40 评估 65.22 秒、峰值 1,291,168,256 bytes；GTSRB A40 评估 177.12 秒、峰值 1,291,160,064 bytes。",
        "成员三收尾：严格 E²LoRA 核心复现 BLOCKED，尚无正式配对结果。EuroSAT 方法核查试跑 job 54504 按论文式 (5)–(7) 得 24 组、秩和 575（LoRA 576），按共享对象去重后 1,325,568 参数，接近 LoRA 的 1,327,104，而非论文显示约 0.66M 与 1.31M。参数数值相近或相符均不单独证明方法正确；文本任务的部分秩更新规则不能直接套用于图像。未为凑预算私自冻结或减秩；证据见 experiments/acheng/reports/core-audit.json。正式配对及依赖其定义的实验因此受阻。",
    ]
    for detail in reversed(details):
        insert_after(body, image_anchor, detail)

    table6 = target_table(body, ["方法", "文本参数", "文本均分", "图像参数", "图像均分"])
    insert_after(body, table6, "成员三收尾：表6图像部分 NOT_STARTED；DoRA、VeRA、LoRI-S 及其 E² 组合尚未实现并验收，不以普通 LoRA 替代。")
    paragraph_after(body, "图 6  Cars 的层间相似度与代理熵", "成员三收尾：图6 NOT_STARTED；Cars 标签映射和训练样本划分尚未验收，梯度诊断链路未就绪，无真实 RMI、代理熵、分组或图像，不填论文数据。")
    table10 = target_table(body, ["路线与搜索", "搜索 ms", "参数", "均分"])
    insert_after(body, table10, "成员三收尾：表10图像贪心与动态规划 NOT_STARTED；相同 RMI、λ、代价及计时口径尚未建立；严格 E²LoRA 方法定义 BLOCKED。文本部分不属于成员三。")
    table11 = target_table(body, ["标准差／强度", "参数", "准确率"])
    insert_after(body, table11, "成员三收尾：表11 NOT_STARTED；未实施分组前加噪或无噪声配对，严格 E²LoRA 方法定义 BLOCKED。")
    table12 = target_table(body, ["种子", "各组层数", "各组秩", "准确率"])
    insert_after(body, table12, "成员三收尾：表12 NOT_STARTED；EuroSAT key 投影种子 0、18、42 的梯度、实际分组、秩及成绩均未运行；严格 E²LoRA 方法定义 BLOCKED。")
    insert_after(body, table13, "成员三收尾：表13 图像 LoRA-r8 的参数复用表3已验收结果，实际整数 1,327,104；七数据集平均准确率 NOT_STARTED，故留空。LoRA-r4 NOT_STARTED；E²LoRA-r8/r16 BLOCKED，未运行。文本部分不属于成员三。")

    # Remove only our additions and restore the one edited cell; the remaining
    # document tree must be identical to the pre-edit original.
    check = copy.deepcopy(root)
    check_body = check.find("w:body", NS)
    notes = [p for p in check_body.xpath("./w:p", namespaces=NS) if text(p).startswith("成员三收尾：")]
    if len(notes) != 12:
        raise ValueError(f"Unexpected annotation count {len(notes)}")
    for note in notes:
        check_body.remove(note)
    check_table = target_table(check_body, ["方法与基础秩", "图像参数", "图像准确率", "文本参数", "文本均分"])
    check_row = next(row for row in check_table.xpath("./w:tr", namespaces=NS)
                     if text(row.xpath("./w:tc", namespaces=NS)[0]) == "LoRA 秩 8")
    check_row.xpath("./w:tc", namespaces=NS)[1].xpath(".//w:t", namespaces=NS)[0].text = "1.31M／____"
    if etree.tostring(check, method="c14n") != etree.tostring(original, method="c14n"):
        raise ValueError("Unexpected modification of existing document content")

    parts["word/document.xml"] = (parts["word/document.xml"][0],
                                  etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True))
    fd, temp = tempfile.mkstemp(prefix=".closing-report-", suffix=".docx", dir=REPORT.parent)
    os.close(fd)
    try:
        with ZipFile(temp, "w") as archive:
            for name, (info, data) in parts.items():
                archive.writestr(info, data)
        with ZipFile(temp) as archive:
            if archive.testzip() is not None:
                raise ValueError("Invalid DOCX archive")
            for name, (_, data) in parts.items():
                if archive.read(name) != data:
                    raise ValueError(f"Unexpected package content: {name}")
            reopened = etree.fromstring(archive.read("word/document.xml"))
            reopened_body = reopened.find("w:body", NS)
            reopened_table = target_table(reopened_body, ["方法与基础秩", "图像参数", "图像准确率", "文本参数", "文本均分"])
            reopened_row = next(row for row in reopened_table.xpath("./w:tr", namespaces=NS)
                                if text(row.xpath("./w:tc", namespaces=NS)[0]) == "LoRA 秩 8")
            if text(reopened_row.xpath("./w:tc", namespaces=NS)[1]) != "1.31M／1.33M":
                raise ValueError("Table 13 reread failed")
            if len([p for p in reopened_body.xpath("./w:p", namespaces=NS) if text(p).startswith("成员三收尾：")]) != 12:
                raise ValueError("Closing notes reread failed")
        os.replace(temp, REPORT)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print("REPORT", REPORT)
    print("BACKUP", backup)
    print("BEFORE_SHA256", original_hash)
    print("AFTER_SHA256", hashlib.sha256(REPORT.read_bytes()).hexdigest())
    print("VERIFIED: six original table-3 scores; table-13 LoRA-r8 parameter; 12 notes; all other existing XML unchanged")


if __name__ == "__main__":
    main()
