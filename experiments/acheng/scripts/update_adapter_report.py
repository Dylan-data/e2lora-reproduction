#!/usr/bin/env python3
"""Update only validated member-three adapter results in the supplied report."""

import argparse
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
DATASET_COUNTS = {"DTD": 1880, "EuroSAT": 2700, "GTSRB": 12630}
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def text(element):
    return "".join(element.xpath(".//w:t/text()", namespaces=NS)).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    result_path = args.result.resolve()
    result_path.relative_to(TASK / "outputs")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_config = json.loads((result_path.parent / "config.json").read_text(encoding="utf-8"))
    dataset_name = run_config["dataset"]["name"]
    pilot = result["status"] == "completed_pilot"
    if not pilot and result["status"] != "completed":
        raise ValueError("Only validated pilot or completed results can be reported.")
    if dataset_name not in DATASET_COUNTS:
        raise ValueError(f"Unsupported dataset for this report updater: {dataset_name}")
    registry = json.loads((TASK / "runs.json").read_text(encoding="utf-8"))
    matches = [r for r in registry["task_jobs"]
               if r.get("experiment_id") == result["experiment_id"] and r.get("job_id") == args.job]
    if len(matches) != 1 or matches[0]["status"] != "COMPLETED_VALIDATED":
        raise ValueError("Job must be uniquely registered and validated before reporting.")
    if pilot:
        if result["adapter_trainable_parameters"] != 718848 or not result["frozen_parameters_unchanged"]:
            raise ValueError("Pilot structure was not validated.")
        note = ("成员三 ShareLoRA 单批验收（非正式准确率）：DTD seed 2026，作业 " + args.job
                + "；每类视觉投影的 A 跨 12 层共享（6 个 A 对象），B 每层独立（72 个 B 对象）。"
                "实际适配及总可训练参数 718,848（0.72M），单批损失 "
                + f"{result['loss']:.4f}，72 个 B 更新，冻结参数不变；B 零初始化使首批 A 梯度为零。"
                "该共享范围是依据论文描述的本组实现选择，非 E²LoRA；无测试成绩。来源："
                + str(result_path.relative_to(ROOT)) + "。")
    else:
        test = result["test"]
        if (result["seed"], result["max_epochs"], test["count"]) != (
            2026, 30, DATASET_COUNTS[dataset_name]
        ):
            raise ValueError("Unexpected formal experiment settings.")
        if sum(a == b for a, b in zip(test["labels"], test["predictions"])) != test["correct"]:
            raise ValueError("Prediction metrics do not match.")
        if result["method"] == "ShareLoRA" and result["rank"] == 8:
            label = "ShareLoRA-r8"
        elif result["method"] == "LoRA" and result["rank"] == 4 and dataset_name == "DTD":
            label = "LoRA-r4"
        else:
            raise ValueError("Unsupported completed result.")
        note = (f"成员三 {dataset_name} {label} seed 2026：作业 {args.job}，30 epochs、"
                f"{result['history'][-1]['global_step']} steps，验证集最佳 epoch {result['best_epoch']}；"
                f"完整测试 {test['correct']}/{test['count']}（{test['accuracy']:.2f}%），"
                f"实际适配及总可训练参数 {result['adapter_trainable_parameters']:,}；"
                f"结果记录 GPU {result['gpu']}、程序计时 {result['gpu_hours']:.3f} GPU-hours，"
                f"Slurm 分段累计占用 {matches[0]['result']['allocated_gpu_seconds'] / 3600:.3f} GPU-hours。"
                "使用该数据集固定镜像划分；单种子，不代表稳定性，未证明与论文划分等价。来源："
                + str(result_path.relative_to(ROOT)) + "。")
    if not args.job.isdigit():
        raise ValueError("Invalid job ID.")
    backup = TASK / "reports" / f"E²LoRA 论文复现报告-{args.job}-更新前-{datetime.now().strftime('%Y%m%dT%H%M%S%z')}.docx"
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(REPORT, backup)
    before_hash = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if hashlib.sha256(backup.read_bytes()).hexdigest() != before_hash:
        raise ValueError("Backup checksum mismatch.")
    with ZipFile(REPORT) as archive:
        parts = {info.filename: (info, archive.read(info.filename)) for info in archive.infolist()}
    root = etree.fromstring(parts["word/document.xml"][1])
    original = copy.deepcopy(root)
    body = root.find("w:body", NS)
    if any(note == text(p) for p in body.xpath("./w:p", namespaces=NS)):
        raise ValueError("This report note already exists.")
    tables = []
    for table in body.xpath("./w:tbl", namespaces=NS):
        rows = table.xpath("./w:tr", namespaces=NS)
        if rows and [text(c) for c in rows[0].xpath("./w:tc", namespaces=NS)] == [
            "方法", "参数", "Cars", "DTD", "EuroSAT", "GTSRB", "RESISC45", "SUN397", "SVHN", "均值"
        ]:
            tables.append(table)
    if len(tables) != 1:
        raise ValueError("Cannot uniquely locate image Table 3.")
    table = tables[0]
    cells = next(row.xpath("./w:tc", namespaces=NS) for row in table.xpath("./w:tr", namespaces=NS)
                 if text(row.xpath("./w:tc", namespaces=NS)[0]) == "ShareLoRA")
    changes = []
    note_changes = []
    if pilot:
        changes.append((cells[1], "0.72M／____", "0.72M／0.72M"))
    elif result["method"] == "ShareLoRA":
        if result["adapter_trainable_parameters"] != 718848:
            raise ValueError("Unexpected ShareLoRA-r8 parameter count.")
        headers = [text(cell) for cell in table.xpath("./w:tr[1]/w:tc", namespaces=NS)]
        if dataset_name not in headers:
            raise ValueError(f"Dataset column is missing: {dataset_name}")
        column = headers.index(dataset_name)
        share_row = next(row for row in table.xpath("./w:tr", namespaces=NS)
                         if text(row.xpath("./w:tc", namespaces=NS)[0]) == "ShareLoRA")
        target = share_row.xpath("./w:tc", namespaces=NS)[column]
        current = text(target)
        if "／" not in current:
            raise ValueError(f"Unexpected result cell: {current!r}")
        paper_value, team_value = current.split("／", 1)
        if team_value != "____":
            raise ValueError(f"Refusing to overwrite an existing report result: {current!r}")
        changes.append((target, current, f"{paper_value}／{result['test']['accuracy']:.2f}"))
    else:
        if result["adapter_trainable_parameters"] != 663552:
            raise ValueError("Unexpected LoRA-r4 parameter count.")
        budget_tables = []
        for candidate in body.xpath("./w:tbl", namespaces=NS):
            rows = candidate.xpath("./w:tr", namespaces=NS)
            if rows and [text(c) for c in rows[0].xpath("./w:tc", namespaces=NS)] == [
                "方法与基础秩", "图像参数", "图像准确率", "文本参数", "文本均分"
            ]:
                budget_tables.append(candidate)
        if len(budget_tables) != 1:
            raise ValueError("Cannot uniquely locate image parameter-budget table.")
        budget_row = next(row for row in budget_tables[0].xpath("./w:tr", namespaces=NS)
                          if text(row.xpath("./w:tc", namespaces=NS)[0]) == "LoRA 秩 4")
        changes.append((budget_row.xpath("./w:tc", namespaces=NS)[1], "0.66M／____", "0.66M／0.66M"))
        status_notes = [p for p in body.xpath("./w:p", namespaces=NS)
                        if text(p).startswith("成员三收尾：表13 图像 LoRA-r8")]
        if len(status_notes) != 1:
            raise ValueError("Cannot uniquely locate prior table-13 status note.")
        status_nodes = status_notes[0].xpath(".//w:t", namespaces=NS)
        if len(status_nodes) != 1 or "LoRA-r4 NOT_STARTED；" not in status_nodes[0].text:
            raise ValueError("Unexpected prior LoRA-r4 status.")
        old_note = status_nodes[0].text
        new_note = old_note.replace("LoRA-r4 NOT_STARTED；", "LoRA-r4 的 DTD 单种子已完成，七项平均尚未完成；")
        status_nodes[0].text = new_note
        note_changes.append((old_note, new_note))
        share_notes = [p for p in body.xpath("./w:p", namespaces=NS)
                       if text(p).startswith("成员三 DTD ShareLoRA-r8 seed 2026：作业 56339")]
        if len(share_notes) != 1:
            raise ValueError("Cannot uniquely locate the earlier ShareLoRA result note.")
        share_nodes = share_notes[0].xpath(".//w:t", namespaces=NS)
        if len(share_nodes) != 1 or "累计 0.526 GPU-hours" not in share_nodes[0].text:
            raise ValueError("Unexpected ShareLoRA timing note.")
        old_share_note = share_nodes[0].text
        share_record = next(r for r in registry["task_jobs"]
                            if r.get("experiment_id") == "table3/dtd/sharelora-r8/seed-2026")
        allocated_hours = share_record["result"]["allocated_gpu_seconds"] / 3600
        new_share_note = old_share_note.replace(
            "累计 0.526 GPU-hours",
            f"程序计时 0.526 GPU-hours、Slurm 分段累计占用 {allocated_hours:.3f} GPU-hours",
        )
        share_nodes[0].text = new_share_note
        note_changes.append((old_share_note, new_share_note))
    for cell, old, new in changes:
        nodes = cell.xpath(".//w:t", namespaces=NS)
        if len(nodes) != 1 or nodes[0].text != old:
            raise ValueError(f"Unexpected report cell: {text(cell)!r}, expected {old!r}")
        nodes[0].text = new
    paragraph = etree.Element(f"{{{W}}}p")
    run = etree.SubElement(paragraph, f"{{{W}}}r")
    node = etree.SubElement(run, f"{{{W}}}t")
    node.text = note
    body.insert(body.index(table) + 1, paragraph)

    check = copy.deepcopy(root)
    cb = check.find("w:body", NS)
    note_nodes = [p for p in cb.xpath("./w:p", namespaces=NS) if text(p) == note]
    if len(note_nodes) != 1:
        raise ValueError("Cannot independently verify added note.")
    cb.remove(note_nodes[0])
    for cell, old, new in changes:
        matching = [n for n in check.xpath("//w:t", namespaces=NS) if n.text == new]
        if len(matching) != 1:
            raise ValueError(f"Updated cell is not unique: {new}")
        matching[0].text = old
    for old_note, new_note in note_changes:
        matching = [n for n in check.xpath("//w:t", namespaces=NS) if n.text == new_note]
        if len(matching) != 1:
            raise ValueError("Updated existing note is not unique.")
        matching[0].text = old_note
    if etree.tostring(check, method="c14n") != etree.tostring(original, method="c14n"):
        raise ValueError("Existing report content unexpectedly changed.")
    parts["word/document.xml"] = (parts["word/document.xml"][0],
                                  etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True))
    fd, temp = tempfile.mkstemp(prefix=".adapter-report-", suffix=".docx", dir=TASK)
    os.close(fd)
    try:
        with ZipFile(temp, "w") as archive:
            for name, (info, data) in parts.items():
                archive.writestr(info, data)
        with ZipFile(temp) as archive:
            if archive.testzip() is not None:
                raise ValueError("Invalid report archive.")
            if any(archive.read(name) != data for name, (_, data) in parts.items()):
                raise ValueError("Report package content mismatch.")
            reopened = etree.fromstring(archive.read("word/document.xml"))
            if len([p for p in reopened.xpath("//w:body/w:p", namespaces=NS) if text(p) == note]) != 1:
                raise ValueError("Report note could not be reread.")
        os.replace(temp, REPORT)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print("REPORT", REPORT, "BACKUP", backup)
    print("CHANGES", [(old, new) for _, old, new in changes])
    print("NOTE", note)
    print("SHA256", hashlib.sha256(REPORT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
