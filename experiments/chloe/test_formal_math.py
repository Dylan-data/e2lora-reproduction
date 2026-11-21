"""Meaningful CPU checks for prompt preservation, scheduling and uploaded eval compatibility."""
import json
from pathlib import Path
import unittest
from formal_math import encode,normalized,lr_multiplier,row_hash
from evaluate import numeric_answer


class Tokenizer:
    eos_token_id=999
    def encode(self,text,add_special_tokens):
        return ([1] if add_special_tokens else [])+[ord(c) for c in text]


class Tests(unittest.TestCase):
    def test_response_preserved(self):
        row=normalized({'query':'  long prompt question   ','response':'  123  '},42)
        x=encode(row,Tokenizer(),10)
        self.assertEqual([v for v in x['labels'] if v!=-100],[49,50,51,999])
        self.assertGreater(x['prompt_trimmed'],0)
        self.assertEqual(len(x['input_ids']),10)
        with self.assertRaises(ValueError):encode(row,Tokenizer(),4)

    def test_scheduler(self):
        self.assertAlmostEqual(lr_multiplier(0,1563,46),1/46)
        self.assertEqual(lr_multiplier(45,1563,46),1)
        self.assertEqual(lr_multiplier(1563,1563,46),0)

    def test_eval_manifest_fixture(self):
        root=Path(__file__).resolve().parent
        fixture=json.loads((root/'configs/member4-eval-fixture.json').read_text(encoding='utf-8'))
        row=fixture['row']
        self.assertEqual(row_hash(row),fixture['sha256'])
        self.assertEqual(numeric_answer(fixture['prediction']),numeric_answer(row['answer'],gold=True))


if __name__=='__main__':unittest.main()
