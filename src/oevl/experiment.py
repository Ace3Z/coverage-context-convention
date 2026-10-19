"""Reproducible Old English length restoration on work-separated source files."""
import argparse,collections,csv,difflib,json,re,unicodedata,statistics,urllib.request,hashlib
from pathlib import Path

LONG='āēīōūȳǣ'; SHORT='aeiouyæ'; TRANS=str.maketrans(dict(zip(LONG,SHORT)))
def norm(s):return unicodedata.normalize('NFC',s).lower()
def bare(s):return norm(s).translate(TRANS)
def key(s):return bare(s).replace('ð','þ').replace('-','')
def pattern(s):return ''.join('1' if c in LONG else '0' for c in norm(s) if c in LONG+SHORT)
def zeros(s):return '0'*len(pattern(s))
def load(path):
 with open(path,encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def tokens(rows):
 for r in rows:
  ts=r['tokens']; bs=r['tokens_bare']
  if len(ts)!=len(bs):raise ValueError('Token count mismatch: '+r['id'])
  for i,(w,b) in enumerate(zip(ts,bs)):
   if bare(w)!=norm(b):raise ValueError('Gold/input mismatch: '+r['id'])
   yield {'id':r['id']+':'+str(i),'sentence_id':r['id'],'source':r['source'],'index':i,'gold':w,'bare':b,'sentence':bs,'gold_pattern':pattern(w),'key':key(b)}
TRAIN=['bright_reader','matthew_ws','sweet_primer','physiologus','zupitza_grammar']
TESTS=['john_ws','soliloquies','tauber_beowulf']  # john_ws is the dev set, scored like the tests
EXCLUDE_D3=('zupitza_grammar',)  # PLAN D3, decided on dev: the final systems train without Zupitza's Grammar
def heldout(root,name,tauber=None):
 # A dev or test set without the sentences that share a 6-gram with train (python -m oevl.audit_splits)
 ex=set(json.loads((Path(root)/'leakage_excluded.json').read_text(encoding='utf8'))['excluded'][name])
 rows=load_tauber(tauber or Path(root)/'brunetti-length.txt') if name=='tauber_beowulf' else list(tokens(load(Path(root)/'by_source'/(name+'.jsonl'))))
 return [r for r in rows if r['sentence_id'] not in ex]
def build_data(root,exclude=()):
 # exclude: training works left out (PLAN D3); the lexicon's training forms and the prior counts follow from train
 src=Path(root)/'by_source'
 train=list(tokens([r for name in TRAIN if name not in exclude for r in load(src/(name+'.jsonl'))]))
 prose=heldout(root,'soliloquies')
 hs=list(tokens(load(src/'hs_beowulf.jsonl'))); wy=list(tokens(load(src/'wyatt_beowulf.jsonl')))
 assert not ({r['source'] for r in train}&{'john_ws','soliloquies','hs_beowulf','wyatt_beowulf'})
 return train,prose,hs,wy

def lexicon(train,external=None):
 # lex[skeleton][pattern] = the lexicon sources that attest it (empty if only training attests it)
 lex=collections.defaultdict(dict); counts=collections.defaultdict(collections.Counter)
 for r in train:
  lex[r['key']].setdefault(r['gold_pattern'],set());counts[r['key']][r['gold_pattern']]+=1
 if external:
  with open(external,encoding='utf8') as f:
   for row in csv.DictReader(f,delimiter='\t' if external.endswith('.tsv') else ','):
    form=row.get('marked_form') or row.get('marked') or row.get('form')
    if form and re.fullmatch(r'[a-zA-ZþðæāēīōūȳǣĀĒĪŌŪȲǢ]+',form):lex[key(form)].setdefault(pattern(form),set()).update((row.get('sources') or 'file').split())
 return lex,counts

def flags(r,lex,counts):
 pats=lex.get(r['key'],set()); ct=counts.get(r['key'],{})
 att=sum(v>=2 for v in ct.values())>=2
 minority=bool(ct.get(r['gold_pattern'],0)) and r['gold_pattern']!=max(ct,key=lambda p:(ct[p],p))
 return {'lexicon_ambiguous':len(pats)>=2,'attested_ambiguous':att,'minority_reading':minority}

def features(r,arm):
 from collections import Counter
 ts=r['sentence'];i=r['index'];w=key(r['bare']);d={'word='+w:1,'prefix='+w[:2]:1,'suffix='+w[-2:]:1}
 if arm in ('left','sentence'):
  for n in (1,2):d[f'L{n}='+ (key(ts[i-n]) if i>=n else '<BOS>')]=1
 if arm=='sentence':
  for n in (1,2):d[f'R{n}='+ (key(ts[i+n]) if i+n<len(ts) else '<EOS>')]=1
 return d

def i32(X):
 # scipy>=1.17 gives int64 sparse indices, which newer scikit-learn SGD rejects; values are unchanged
 X.indices=X.indices.astype('int32');X.indptr=X.indptr.astype('int32');return X
def predict(train,test,lex,counts,arm,seed=42):
 if arm=='frequency':
  # PLAN S8: most frequent in train; for a skeleton unseen in train, the pattern most lexicon sources attest; then fewest long
  return [max(lex.get(r['key'],{zeros(r['gold'])}),key=lambda p:(counts[r['key']][p],0 if counts[r['key']] else len(lex[r['key']][p]),-p.count('1'),p)) if lex.get(r['key']) else '0'*len(r['gold_pattern']) for r in test]
 from sklearn.feature_extraction import DictVectorizer
 from sklearn.linear_model import SGDClassifier
 # Learn length patterns for observed tokens; candidate constraints at inference.
 eligible=[r for r in train if len(r['gold_pattern'])>0 and len(lex.get(r['key'],set()))>=2]
 vec=DictVectorizer();X=i32(vec.fit_transform([features(r,arm) for r in eligible])); y=[r['gold_pattern'] for r in eligible]
 model=SGDClassifier(loss='log_loss',max_iter=35,tol=1e-3,random_state=seed);model.fit(X,y)
 probs=model.predict_proba(i32(vec.transform([features(r,arm) for r in test])));classes=list(model.classes_)
 out=[]
 for r,ps in zip(test,probs):
  opts=lex.get(r['key'],set()); fallback=zeros(r['gold'])
  opts=[p for p in opts if len(p)==len(fallback)]
  out.append(max(opts,key=lambda p:(ps[classes.index(p)] if p in classes else 0,counts[r['key']][p])) if opts else fallback)
 return out

def load_classes(path):
 with open(path,encoding='utf8') as f:return {r['skeleton']:r['final'] for r in csv.DictReader(f)}

def evaluate(test,preds,lex,counts,cls=None):
 out={'n':len(test),'accuracy':sum(p==r['gold_pattern'] for r,p in zip(test,preds))/len(test)}
 for label in ['lexicon_ambiguous','attested_ambiguous','minority_reading']:
  ix=[i for i,r in enumerate(test) if flags(r,lex,counts)[label]]
  out[label]={'n':len(ix),'accuracy':sum(preds[i]==test[i]['gold_pattern'] for i in ix)/len(ix) if ix else None}
 # S6 classes split the lexicon-ambiguous tokens; 'unlisted' are ambiguous skeletons outside the class file
 for c in (['A','B','C','unlisted'] if cls else []):
  ix=[i for i,r in enumerate(test) if len(lex.get(r['key'],()))>=2 and cls.get(r['key'],'unlisted')==c]
  out['class_'+c]={'n':len(ix),'accuracy':sum(preds[i]==test[i]['gold_pattern'] for i in ix)/len(ix) if ix else None}
 return out

def oracle(test,lex):
 # Oracle must use the same OOV fallback as the actual systems.
 return [r['gold_pattern'] if r['gold_pattern'] in lex.get(r['key'],set()) else '0'*len(r['gold_pattern']) for r in test]

def mcnemar(a,b,gold):
 from scipy.stats import binomtest
 b01=sum(x!=g and y==g for x,y,g in zip(a,b,gold));b10=sum(x==g and y!=g for x,y,g in zip(a,b,gold));n=b01+b10
 return {'a_wrong_b_right':b01,'a_right_b_wrong':b10,'discordant':n,'p_exact':binomtest(min(b01,b10),n,0.5).pvalue if n else 1.0}

def agreement(a,b,lex):
 # Sequence alignment on unmarked tokens; include alignment rate.
 ak=[r['key'] for r in a];bk=[r['key'] for r in b]
 sm=difflib.SequenceMatcher(None,ak,bk,autojunk=False); pairs=[]
 for op,i,j,k,l in sm.get_opcodes():
  if op=='equal':pairs.extend(zip(a[i:j],b[k:l]))
 from sklearn.metrics import cohen_kappa_score
 result={'aligned':len(pairs),'a_total':len(a),'b_total':len(b),'alignment_rate_a':len(pairs)/len(a)}
 for label,subset in [('all',pairs),('lexicon_ambiguous',[(x,y) for x,y in pairs if len(lex.get(x['key'],set()))>=2])]:
  aa=[x['gold_pattern'] for x,y in subset];bb=[y['gold_pattern'] for x,y in subset]
  result[label]={'n':len(subset),'agreement':sum(x==y for x,y in zip(aa,bb))/len(aa) if aa else None,'cohen_kappa':float(cohen_kappa_score(aa,bb)) if aa and len(set(aa+bb))>1 else None}
 return result

def clean_tauber(s):
 s=unicodedata.normalize('NFD',s).replace('\u0307','').replace('\u0328','').replace('\u0302','\u0304')
 return unicodedata.normalize('NFC',s).replace('ċ','c').replace('ġ','g')

def load_tauber(path):
 sentences=[];ts=[];bs=[];sid=0;total=0;marks=0
 for line in Path(path).read_text(encoding='utf8').splitlines():
  if not line.strip():continue
  cols=line.split('|')
  if len(cols)!=18:raise ValueError(f'Tauber column count {len(cols)} at row {total+1}')
  b=clean_tauber(cols[9]);w=clean_tauber(cols[-1]);punct=cols[10]
  if bare(w)!=norm(b):raise ValueError(f'Tauber marked/bare mismatch row {total+1}: {b!r} {w!r}')
  bw,ww=b.split(),w.split()
  if len(bw)!=len(ww):raise ValueError(f'Tauber word-count mismatch row {total+1}: {b!r}')
  for b1,w1 in zip(bw,ww):ts.append(w1);bs.append(b1);total+=1;marks+=sum(c in LONG for c in norm(w1))
  if any(c in punct for c in '.!?'):
   sentences.append({'id':f'tauber:{sid}','source':'tauber_beowulf','tokens':ts,'tokens_bare':bs});ts=[];bs=[];sid+=1
 if ts:sentences.append({'id':f'tauber:{sid}','source':'tauber_beowulf','tokens':ts,'tokens_bare':bs})
 result=list(tokens(sentences))
 if len(result)!=total:raise AssertionError('Tauber token loss')
 print('Tauber validated:',total,'tokens,',marks,'long marks')
 return result

def per_word(train,test,pred,lex,counts,top=20,cls=None):
 subset=[(i,r) for i,r in enumerate(test) if flags(r,lex,counts)['lexicon_ambiguous']]
 most=collections.Counter(r['key'] for _,r in subset).most_common(top)
 out=[]
 for word,n in most:
  indices=[i for i,r in subset if r['key']==word];gold=[test[i]['gold_pattern'] for i in indices]
  row={'word':word,'n':n,'category_preliminary':(cls or {}).get(word,'unlisted'),'test_pattern_counts':json.dumps(dict(collections.Counter(gold)),ensure_ascii=False)}
  for name,values in pred.items():row[name+'_accuracy']=round(sum(values[i]==test[i]['gold_pattern'] for i in indices)/n,6)
  for source in ['bright_reader','matthew_ws','sweet_primer','physiologus','zupitza_grammar']:
   ct=collections.Counter(r['gold_pattern'] for r in train if r['source']==source and r['key']==word)
   row[source+'_pattern_counts']=json.dumps(dict(ct),ensure_ascii=False)
  out.append(row)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--out',default='results/experiment');p.add_argument('--tauber',default='data/brunetti-length.txt');p.add_argument('--lexicon',help='Optional externally sourced lexicon TSV/CSV');p.add_argument('--tests',default='soliloquies,tauber_beowulf',help='Comma list: john_ws (dev), soliloquies, tauber_beowulf (run one at a time to keep each run short)');p.add_argument('--exclude-train',default='',help='Comma list of training works to leave out, e.g. zupitza_grammar');a=p.parse_args()
 wanted=set(a.tests.split(','));exclude=[x for x in a.exclude_train.split(',') if x]
 if wanted-set(TESTS):p.error(f'unknown --tests {sorted(wanted-set(TESTS))}: use {",".join(TESTS)}')
 if set(exclude)-set(TRAIN):p.error(f'unknown --exclude-train {sorted(set(exclude)-set(TRAIN))}: use {",".join(TRAIN)}')
 cpath=Path(a.data)/'ambiguity_classes.csv'  # optional: without it no per-class accuracy is reported
 train,prose,hs,wy=build_data(a.data,exclude);lex,counts=lexicon(train,a.lexicon);cls=load_classes(cpath) if cpath.exists() else None;out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 summary={'dataset':{'train':len(train),'test_prose':len(prose),'agreement_hs':len(hs),'agreement_wyatt':len(wy),'lexicon_origin':'training attested forms'+(' + external file' if a.lexicon else ' (proxy; NOT Clark Hall/Wiktionary/Bosworth-Toller)')}}
 if exclude:summary['dataset']['exclude_train']=exclude  # only when set: the default metrics.json stays as before
 mpath=out/'metrics.json'  # refuse a mismatched --out folder before any file in it is overwritten
 if mpath.exists() and any(json.loads(mpath.read_text(encoding='utf8')).get('dataset',{}).get(k)!=summary['dataset'].get(k) for k in ('lexicon_origin','exclude_train')):raise SystemExit(f'{mpath} was made with a different lexicon or training set; use another --out folder')
 tauber_path=Path(a.tauber)
 if 'tauber_beowulf' in wanted and not tauber_path.exists():raise SystemExit('Tauber gold missing: put brunetti-length.txt in data/ or run python -m oevl.fetch_tauber')
 tests=[(name,heldout(a.data,name,tauber_path)) for name in TESTS if name in wanted]
 for name,test in tests:
  pred={'frequency':predict(train,test,lex,counts,'frequency'),'oracle':oracle(test,lex)}
  for arm in ['word','left','sentence']:
   for seed in (1,2,3):pred[f'{arm}_seed{seed}']=predict(train,test,lex,counts,arm,seed)
  summary[name]={arm:evaluate(test,values,lex,counts,cls) for arm,values in pred.items()}
  summary[name]['seed_summary']={}
  for arm in ['word','left','sentence']:
   summary[name]['seed_summary'][arm]={}
   for label in ['accuracy','lexicon_ambiguous','attested_ambiguous','minority_reading']+(['class_A','class_B','class_C','class_unlisted'] if cls else []):
    vals=[summary[name][f'{arm}_seed{s}']['accuracy' if label=='accuracy' else label]['accuracy'] if label!='accuracy' else summary[name][f'{arm}_seed{s}']['accuracy'] for s in (1,2,3)]
    summary[name]['seed_summary'][arm][label]={'mean':statistics.mean(vals) if all(v is not None for v in vals) else None,'sd':statistics.stdev(vals) if all(v is not None for v in vals) else None}
  gold=[r['gold_pattern'] for r in test]
  pairs=[('frequency',f'{arm}_seed1') for arm in ['word','left','sentence']]+[(f'{x}_seed1',f'{y}_seed1') for x,y in [('word','left'),('word','sentence'),('left','sentence')]]
  summary[name]['mcnemar_seed1']={f'{x}_vs_{y}':mcnemar(pred[x],pred[y],gold) for x,y in pairs}
  summary[name]['per_word']=per_word(train,test,pred,lex,counts,cls=cls)
  with open(out/(name+'_predictions.jsonl'),'w',encoding='utf8') as f:
   for i,r in enumerate(test):f.write(json.dumps({**{k:v for k,v in r.items() if k!='sentence'},**flags(r,lex,counts),**{arm+'_prediction':values[i] for arm,values in pred.items()}},ensure_ascii=False)+'\n')
  with open(out/(name+'_per_word.csv'),'w',encoding='utf8',newline='') as f:
   rows=summary[name]['per_word'];writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['word']);writer.writeheader();writer.writerows(rows)
 summary['editor_agreement']=agreement(hs,wy,lex)
 summary['statistical_test']='Exact paired McNemar, seed 1, all tokens; observations within sentences are not independent.'
 summary['model_seeds']=[1,2,3]
 if mpath.exists():  # runs with --tests one at a time add their test to the same file
  merged=json.loads(mpath.read_text(encoding='utf8'));merged.update(summary);summary=merged
 mpath.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
