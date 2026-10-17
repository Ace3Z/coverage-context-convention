import unittest,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import clean_tauber,load_tauber,bare,pattern
from wikipron_to_marked import mark
ROOT=Path(__file__).resolve().parents[1]
class TestV4(unittest.TestCase):
 def test_wikipron(self):
  for w,ipa,expected in [('god','ɡ oː d','gōd'),('eage','æ͜ɑː ɡ e','ēage'),('þeostrian','θ e͜oː s t r i ɑ n','þēostrian'),('ceald','t͡ʃ æ͜ɑ l d','ceald')]:self.assertEqual(mark(w,ipa),expected)
 def test_circumflex(self):self.assertEqual(clean_tauber('gân'),'gān');self.assertEqual(clean_tauber('dôn'),'dōn')
 def test_multiword(self):
  cols=['']*18;cols[9]='oð þæt';cols[10]='.';cols[-1]='oð þæt'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'sample.txt';p.write_text('|'.join(cols)+'\n',encoding='utf-8')
   self.assertEqual(len(load_tauber(p)),2)
 @unittest.skipUnless((ROOT/'data/brunetti-length.txt').exists(),'Real Tauber source not downloaded')
 def test_real_tauber(self):
  rows=load_tauber(ROOT/'data/brunetti-length.txt')
  self.assertEqual(len(rows),17361)
  self.assertEqual(sum(p.count('1') for p in (r['gold_pattern'] for r in rows)),7438)
if __name__=='__main__':unittest.main()
