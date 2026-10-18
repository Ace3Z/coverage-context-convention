import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from oevl.experiment import *
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
 def test_prior_backoff_to_lexicon_sources(self):
  import tempfile
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'lex.csv';p.write_text('marked_form,sources\ngōd,clarkhall bt\ngod,wiktionary\nmān,bt\nman,wiktionary\nstān,bt\n',encoding='utf8')
   train=[{'key':'stan','gold_pattern':'0'}]*2  # seen in train: training counts win over the lexicon
   lex,counts=lexicon(train,str(p))
  test=[{'key':k,'gold':k,'gold_pattern':zeros(k)} for k in ('god','man','stan','wæter')]
  # unseen: most sources, then fewest long (tie), seen, out of lexicon -> all short
  self.assertEqual(predict(train,test,lex,counts,'frequency'),['1','0','0','00'])

class ClassTests(unittest.TestCase):
 def test_class_accuracy(self):
  lex={'a':{'0','1'},'b':{'0','1'},'c':{'0'}};cts=collections.defaultdict(collections.Counter)
  test=[{'key':'a','gold_pattern':'1'},{'key':'a','gold_pattern':'0'},{'key':'b','gold_pattern':'0'},{'key':'c','gold_pattern':'0'}]
  out=evaluate(test,['1','1','0','0'],lex,cts,{'a':'C','c':'A'})
  self.assertEqual(out['class_C'],{'n':2,'accuracy':0.5});self.assertEqual(out['class_A'],{'n':0,'accuracy':None})
  self.assertEqual(out['class_unlisted'],{'n':1,'accuracy':1.0})
  self.assertEqual(sum(out['class_'+c]['n'] for c in ['A','B','C','unlisted']),out['lexicon_ambiguous']['n'])
  self.assertNotIn('class_A',evaluate(test,['1','1','0','0'],lex,cts))
 def test_class_file(self):
  from sklearn.metrics import cohen_kappa_score
  d=Path(__file__).resolve().parents[1]/'data'
  rows=list(csv.DictReader(open(d/'ambiguity_classes.csv',encoding='utf8')));meta=json.loads((d/'ambiguity_classes.json').read_text(encoding='utf8'))
  a=[r['label_a'] for r in rows];b=[r['label_b'] for r in rows];final=[r['final'] for r in rows]
  self.assertEqual(len(rows),meta['n_skeletons']);self.assertEqual(load_classes(d/'ambiguity_classes.csv'),dict(zip([r['skeleton'] for r in rows],final)))
  self.assertAlmostEqual(cohen_kappa_score(a,b),meta['cohen_kappa'],places=6);self.assertAlmostEqual(sum(x==y for x,y in zip(a,b))/len(a),meta['agreement'],places=6)
  self.assertEqual({c:final.count(c) for c in 'ABC'},meta['counts']['final'])
  self.assertTrue(all(f in ('A','B','C') and (f==x==y or (x!=y and r['adjudication_note']!='agreed')) for r,f,x,y in zip(rows,final,a,b)))
