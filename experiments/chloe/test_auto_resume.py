import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from auto_resume import retryable,initial,tick


class Tests(unittest.TestCase):
    def test_retry_scope(self):
        for state in ['TIMEOUT','PREEMPTED','NODE_FAIL','BOOT_FAIL']:
            self.assertTrue(retryable(state,None))
        self.assertTrue(retryable('FAILED','75:0'))
        for state,code in [('FAILED','1:0'),('CANCELLED','0:0'),('OUT_OF_MEMORY','0:125')]:
            self.assertFalse(retryable(state,code))

    def test_active_jobs_never_resubmitted(self):
        state=initial();commands=[]
        def execute(args):commands.append(args);return ''
        def query(job_id,_):return ('RUNNING' if job_id=='56320' else 'PENDING'),None
        tick(state,lambda s:None,execute,query)
        self.assertFalse(commands)

    def test_repair_dependency_without_submission(self):
        state=initial();state['runs'][0]['job_id']='60000';commands=[]
        def execute(args):commands.append(args);return ''
        def query(job_id,_):return ('RUNNING' if job_id=='60000' else 'PENDING'),None
        tick(state,lambda s:None,execute,query)
        self.assertEqual(commands,[['scontrol','update','JobId=56321','Dependency=afterok:60000']])
        self.assertEqual(state['runs'][1]['depends_on'],'60000')

    def test_timeout_resumes_and_relinks_successor(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'checkpoint.pt').write_bytes(b'fixture')
            state=initial();commands=[]
            def execute(args):
                commands.append(args)
                return '60001' if args[0]=='sbatch' else ''
            def query(job_id,_):return ('TIMEOUT','0:0') if job_id=='56320' else ('PENDING',None)
            with patch('auto_resume.directory',return_value=root),patch('auto_resume.verify_sources'):
                tick(state,lambda s:None,execute,query)
            self.assertEqual(state['runs'][0]['job_id'],'60001')
            self.assertEqual(state['runs'][0]['resumes'],1)
            self.assertEqual(commands[-1],['scontrol','update','JobId=56321','Dependency=afterok:60001'])
            self.assertEqual(state['runs'][1]['depends_on'],'60001')


if __name__=='__main__':unittest.main()
