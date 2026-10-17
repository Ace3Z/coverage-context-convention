import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import *
class Tests(unittest.TestCase):
 def test_macron(self):self.assertEqual(bare('ǣ'), 'æ');self.assertEqual(pattern('gōd'),'1')
 def test_oracle_oov(self):
  rs=[{'key':'x','gold_pattern':'1'},{'key':'y','gold_pattern':'0'},{'key':'z','gold_pattern':'1'}]
  self.assertEqual(oracle(rs,{'y':{'0'},'z':{'0'}}),['0','0','0'])
 def test_ambiguity(self):
  lex={'a':{'0','1'}};cts=collections.defaultdict(collections.Counter,{'a':collections.Counter({'0':8,'1':2})})
  self.assertEqual(flags({'key':'a','gold_pattern':'1'},lex,cts),{'lexicon_ambiguous':True,'attested_ambiguous':True,'minority_reading':True})
 def test_mcnemar(self):self.assertEqual(mcnemar(['0'],['1'],['1'])['p_exact'],1.0)
 def test_source_split(self):
  tr,pr,hs,wy=build_data(str(Path(__file__).resolve().parents[1]/'data'))
  self.assertTrue(tr and pr and hs and wy);self.assertNotIn('soliloquies',{r['source'] for r in tr})
if __name__=='__main__':unittest.main()

class RegressionTests(unittest.TestCase):
 def test_full_training_no_cap(self):
  train,_,_,_=build_data(str(Path(__file__).resolve().parents[1]/'data'))
  lex,counts=lexicon(train)
  eligible=[r for r in train if len(r['gold_pattern'])>0 and len(lex.get(r['key'],set()))>=2]
  self.assertGreater(len(eligible),18000)
  self.assertEqual(len({r['source'] for r in eligible}),5)
 def test_tauber_parser(self):
  import tempfile
  cols=['']*18;cols[9]='god';cols[10]='.';cols[-1]='gōd'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'gold.txt';p.write_text('|'.join(cols)+'\n',encoding='utf8')
   rows=load_tauber(p)
   self.assertEqual(rows[0]['gold_pattern'],'1')
 def test_autojunk_alignment(self):
  import inspect
  self.assertIn('autojunk=False',inspect.getsource(agreement))
