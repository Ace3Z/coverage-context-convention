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
 def test_exclude_train(self):
  import os,subprocess,sys
  root=Path(__file__).resolve().parents[1];tr,_,_,_=build_data(str(root/'data'),('zupitza_grammar',))
  self.assertEqual({r['source'] for r in tr},set(TRAIN)-{'zupitza_grammar'})
  p=subprocess.run([sys.executable,'-m','oevl.experiment','--exclude-train','zupitza'],env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
  self.assertEqual(p.returncode,2,p.stderr)
 def test_excluded_work_never_reaches_training(self):
  # PLAN D3, as run_all.sh runs it: only Zupitza marks 'god' long, so left out it adds no token, training form or prior count
  import os,subprocess,sys,tempfile
  root=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);(d/'by_source').mkdir()
   for name in TRAIN+['soliloquies','hs_beowulf','wyatt_beowulf']:
    ws=['gōd']*5 if name=='zupitza_grammar' else ['god','man','mān']
    (d/'by_source'/(name+'.jsonl')).write_text(json.dumps({'id':name+':0','source':name,'tokens':ws,'tokens_bare':[bare(w) for w in ws]},ensure_ascii=False)+'\n',encoding='utf8')
   (d/'leakage_excluded.json').write_text(json.dumps({'excluded':{t:[] for t in TESTS}}),encoding='utf8')
   train=build_data(d,EXCLUDE_D3)[0];lex,counts=lexicon(train)
   self.assertEqual(({r['source'] for r in train}&set(EXCLUDE_D3),set(lex['god']),counts['god']),(set(),{'0'},{'0':4}))
   for out,flag,want in (('d3','zupitza_grammar',(12,False,'0')),('all5','',(17,True,'1'))):  # all5: the fixture can fail
    p=subprocess.run([sys.executable,'-m','oevl.experiment','--data',str(d),'--out',str(d/out),'--tests','soliloquies','--exclude-train',flag],env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
    self.assertEqual(p.returncode,0,p.stderr)
    god=json.loads((d/out/'soliloquies_predictions.jsonl').read_text(encoding='utf8').splitlines()[0])
    self.assertEqual((json.loads((d/out/'metrics.json').read_text(encoding='utf8'))['dataset']['train'],god['lexicon_ambiguous'],god['frequency_prediction']),want)
   before=(d/'d3'/'soliloquies_predictions.jsonl').read_bytes()  # a run without the D3 flag into the D3 folder is refused before it writes
   p=subprocess.run([sys.executable,'-m','oevl.experiment','--data',str(d),'--out',str(d/'d3'),'--tests','soliloquies'],env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
   self.assertEqual((p.returncode,(d/'d3'/'soliloquies_predictions.jsonl').read_bytes()),(1,before),p.stderr)
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
  with open(d/'ambiguity_classes.csv',encoding='utf8') as f:rows=list(csv.DictReader(f))
  meta=json.loads((d/'ambiguity_classes.json').read_text(encoding='utf8'))
  a=[r['label_a'] for r in rows];b=[r['label_b'] for r in rows];final=[r['final'] for r in rows]
  self.assertEqual(len(rows),meta['n_skeletons']);self.assertEqual(load_classes(d/'ambiguity_classes.csv'),dict(zip([r['skeleton'] for r in rows],final)))
  self.assertAlmostEqual(cohen_kappa_score(a,b),meta['cohen_kappa'],places=6);self.assertAlmostEqual(sum(x==y for x,y in zip(a,b))/len(a),meta['agreement'],places=6)
  self.assertEqual({c:final.count(c) for c in 'ABC'},meta['counts']['final'])
  self.assertTrue(all(f in ('A','B','C') and (f==x==y or (x!=y and r['adjudication_note']!='agreed')) for r,f,x,y in zip(rows,final,a,b)))
 def test_class_file_d3(self):
  d=Path(__file__).resolve().parents[1]/'data';d3=json.loads((d/'ambiguity_classes.json').read_text(encoding='utf8'))['d3']
  listed=load_classes(d/'ambiguity_classes.csv');train=build_data(str(d),EXCLUDE_D3)[0]
  toks=train+list(tokens(load(d/'by_source'/'soliloquies.jsonl')))+load_tauber(d/'brunetti-length.txt')
  self.assertEqual(len(toks),d3['tokens_counted'])
  for name,ext in (('training_and_wikipron',str(d/'lex_wikipron.csv')),('training_only',None)):
   lex=lexicon(train,ext)[0];amb=[r['key'] for r in toks if len(lex.get(r['key'],()))>=2]
   self.assertEqual({'listed_not_ambiguous':[k for k in listed if len(lex.get(k,()))<2],'ambiguous_tokens':len(amb),'ambiguous_tokens_outside_list':sum(k not in listed for k in amb)},d3[name])
 def test_per_word_class(self):
  lex={'ge':{'0':set(),'1':set()},'x':{'0':set(),'1':set()}};cts=collections.defaultdict(collections.Counter)
  rows=per_word([],[{'key':'ge','gold_pattern':'1'},{'key':'x','gold_pattern':'0'}],{'frequency':['1','0']},lex,cts,cls={'ge':'B'})
  self.assertEqual({r['word']:r['category_preliminary'] for r in rows},{'ge':'B','x':'unlisted'})
