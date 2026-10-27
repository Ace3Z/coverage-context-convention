"""Reproducible Old English length restoration on work-separated source files.

Runs the frequency prior, the oracle and the logistic word, left and bilateral models (arm "sentence" here), each
choosing among a key's candidate length patterns: the training forms plus an optional --lexicon file. Training is
the works in TRAIN minus --exclude-train, and every set in --tests is scored without the sentences that share a word
6-gram with any TRAIN work (heldout). It also measures length agreement between the Harrison and Sharp and the Wyatt
editions of Beowulf.

  python -m oevl.experiment --data data --out results/wikipron --lexicon data/lex_wikipron.csv
      --exclude-train zupitza_grammar --tests soliloquies
  -> results/wikipron/metrics.json, soliloquies_predictions.jsonl and soliloquies_per_word.csv
"""
import argparse,collections,csv,difflib,hashlib,json,re,unicodedata,statistics
from pathlib import Path
from typing import TYPE_CHECKING,Any
from collections.abc import Collection,Iterable,Iterator,Mapping,Sequence
if TYPE_CHECKING:from scipy.sparse import csr_matrix

# the vowel letters with and without macron, position for position, and the table that removes the macron
LONG='āēīōūȳǣ'; SHORT='aeiouyæ'; TRANS=str.maketrans(dict(zip(LONG,SHORT)))
def norm(s:str)->str:
 """NFC-normalise s and lower-case it."""
 return unicodedata.normalize('NFC',s).lower()
def bare(s:str)->str:
 """s normalised by norm, with every macron vowel of LONG written as its short letter."""
 return norm(s).translate(TRANS)
def key(s:str)->str:
 """A word's lookup key, or skeleton: its bare form with ð written þ and hyphens and apostrophes (' and ’) dropped."""
 return bare(s).replace('ð','þ').replace('-','').replace('’','').replace("'",'')
def pattern(s:str)->str:
 """The length pattern of s: one digit per vowel letter (a e i o u y æ), 1 if it bears a macron, else 0.

 Other letters are skipped, so a word without vowels gives '' and an unmarked word gives all zeros."""
 return ''.join('1' if c in LONG else '0' for c in norm(s) if c in LONG+SHORT)
def zeros(s:str)->str:
 """The all-short pattern of s: '0' for each of its vowel letters."""
 return '0'*len(pattern(s))
def load(path:str|Path)->list[dict[str,Any]]:
 """Read a UTF-8 JSON Lines file into a list of dicts, skipping blank lines."""
 with open(path,encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def tokens(rows:Iterable[Mapping[str,Any]])->Iterator[dict[str,Any]]:
 """Yield one token dict per word of each sentence record, after checking that its marked and bare forms match.

 rows: sentence dicts with id, source, tokens (length-marked) and tokens_bare (unmarked), as in by_source/*.jsonl.
 Each token dict has id ('<sentence id>:<index>'), sentence_id, source, index, gold (the marked word), bare (the
 unmarked word as stored), sentence (the sentence's tokens_bare list, shared by all its tokens), gold_pattern
 (pattern of gold) and key (key of bare). Raises ValueError when the two token lists of a sentence differ in length or
 when bare(gold) != norm(bare) for a word."""
 for r in rows:
  ts=r['tokens']; bs=r['tokens_bare']
  if len(ts)!=len(bs):raise ValueError('Token count mismatch: '+r['id'])
  for i,(w,b) in enumerate(zip(ts,bs)):
   if bare(w)!=norm(b):raise ValueError('Gold/input mismatch: '+r['id'])
   yield {'id':r['id']+':'+str(i),'sentence_id':r['id'],'source':r['source'],'index':i,'gold':w,'bare':b,'sentence':bs,'gold_pattern':pattern(w),'key':key(b)}
# the five training works (splits are by work, so no dev or test work is seen in training)
TRAIN=['bright_reader','matthew_ws','sweet_primer','physiologus','zupitza_grammar']
TESTS=['john_ws','soliloquies','tauber_beowulf']  # john_ws is the dev set, scored like the tests
EXCLUDE_D3=('zupitza_grammar',)  # decision D3, decided on dev: the final systems train without Zupitza's Grammar
def heldout(root:str|Path,name:str,tauber:str|Path|None=None)->list[dict[str,Any]]:
 """A dev or test set as token dicts, without the sentences that share a word 6-gram with training.

 root: the data folder. name: a key of root/leakage_excluded.json's 'excluded' map (john_ws, soliloquies or
 tauber_beowulf), whose sentence ids are dropped. python -m oevl.audit_splits writes that file against all five TRAIN
 works, so Zupitza's 6-grams are excluded even when D3 leaves the work out of training. tauber_beowulf is read with
 load_tauber from tauber, or from root/brunetti-length.txt when tauber is None. Any other set is read from
 root/by_source/<name>.jsonl."""
 ex=set(json.loads((Path(root)/'leakage_excluded.json').read_text(encoding='utf8'))['excluded'][name])
 rows=load_tauber(tauber or Path(root)/'brunetti-length.txt') if name=='tauber_beowulf' else list(tokens(load(Path(root)/'by_source'/(name+'.jsonl'))))
 return [r for r in rows if r['sentence_id'] not in ex]
def build_data(root:str|Path,exclude:Collection[str]=()
               )->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
 """Load the training tokens, the prose test set and the two Beowulf agreement editions from the data folder root.

 exclude: works of TRAIN to leave out of training (decision D3 leaves out EXCLUDE_D3). As the lexicon's training forms
 and the prior's counts are built from train, a left-out work adds neither. Names not in TRAIN are ignored here.
 Returns (train, prose, hs, wy), lists of token dicts: the kept TRAIN works, heldout(root, 'soliloquies'), and the
 whole hs_beowulf and wyatt_beowulf files. Asserts that no training token comes from John, the Soliloquies or the
 two agreement editions."""
 src=Path(root)/'by_source'
 train=list(tokens([r for name in TRAIN if name not in exclude for r in load(src/(name+'.jsonl'))]))
 prose=heldout(root,'soliloquies')
 hs=list(tokens(load(src/'hs_beowulf.jsonl'))); wy=list(tokens(load(src/'wyatt_beowulf.jsonl')))
 assert not ({r['source'] for r in train}&{'john_ws','soliloquies','hs_beowulf','wyatt_beowulf'})
 return train,prose,hs,wy

def lexicon(train:Iterable[Mapping[str,Any]],external:str|None=None)->tuple[
            collections.defaultdict[str,dict[str,set[str]]],collections.defaultdict[str,collections.Counter[str]]]:
 """Build the candidate lexicon and the prior's training counts, both keyed by key().

 train: token dicts, read through key and gold_pattern. external: an optional path, as a str, to a CSV file, or a
 tab-separated one when the name ends in .tsv, whose form is the first non-empty of the marked_form, marked and form
 columns. A row is skipped when its form is empty or has a character outside a-z, A-Z, lower-case þ, ð and æ and the
 precomposed macron vowels of either case. The form is not normalised first, so a hyphen, a capital Þ, Ð or Æ, or a
 decomposed macron also drops the row.
 Returns (lex, counts), two defaultdicts. lex[key][pattern] is the set of lexicon sources that list the pattern,
 from the space-separated sources column or 'file' when that is empty or missing, and an empty set when only
 training attests it. counts[key] is a Counter of the gold patterns of that key in train."""
 lex:collections.defaultdict[str,dict[str,set[str]]];counts:collections.defaultdict[str,collections.Counter[str]]
 lex=collections.defaultdict(dict); counts=collections.defaultdict(collections.Counter)
 for r in train:
  lex[r['key']].setdefault(r['gold_pattern'],set());counts[r['key']][r['gold_pattern']]+=1
 if external:
  with open(external,encoding='utf8') as f:
   for row in csv.DictReader(f,delimiter='\t' if external.endswith('.tsv') else ','):
    form=row.get('marked_form') or row.get('marked') or row.get('form')
    if form and re.fullmatch(r'[a-zA-ZþðæāēīōūȳǣĀĒĪŌŪȲǢ]+',form):lex[key(form)].setdefault(pattern(form),set()).update((row.get('sources') or 'file').split())
 return lex,counts

def flags(r:Mapping[str,Any],lex:Mapping[str,Collection[str]],counts:Mapping[str,Mapping[str,int]])->dict[str,bool]:
 """The three harder-population flags of token r (read through key and gold_pattern), as a dict of booleans.

 lexicon_ambiguous: the key has two or more candidates in lex. attested_ambiguous: at least two of its patterns are
 each seen at least twice in training. minority_reading: the gold pattern is seen in training, but strictly less
 often than the key's most frequent training pattern."""
 pats=lex.get(r['key'],set()); ct=counts.get(r['key'],{})
 att=sum(v>=2 for v in ct.values())>=2
 # strict: the gold occurs in training, but less often than the most frequent pattern (a tied pattern is no minority)
 minority=0<ct.get(r['gold_pattern'],0)<max(ct.values())
 return {'lexicon_ambiguous':len(pats)>=2,'attested_ambiguous':att,'minority_reading':minority}

def features(r:Mapping[str,Any],arm:str)->dict[str,int]:
 """Sparse indicator features of token r for the logistic models, as a dict of feature name -> 1.

 r is read through sentence, index and bare, and every word is used as its key. Every arm has the word and its first
 and last two letters. 'left' adds the two preceding words, 'sentence' (the paper's bilateral) also the two following
 words, with <BOS> or <EOS> past the sentence edge. 'conjoined' adds to the sentence features a copy of each
 neighbour tied to the word, written W|<slot>=<word>|<neighbour>, such as W|L1=þe|to for þe after to. Any other arm,
 such as 'word', gets the word features only."""
 ts=r['sentence'];i=r['index'];w=key(r['bare']);d={'word='+w:1,'prefix='+w[:2]:1,'suffix='+w[-2:]:1}
 if arm in ('left','sentence','conjoined'):
  for n in (1,2):d[f'L{n}='+ (key(ts[i-n]) if i>=n else '<BOS>')]=1
 if arm in ('sentence','conjoined'):
  for n in (1,2):d[f'R{n}='+ (key(ts[i+n]) if i+n<len(ts) else '<EOS>')]=1
 if arm=='conjoined':  # post hoc diagnostic (oevl.paper_numbers), not run by main(): each neighbour also tied to the word
  d.update({f'W|{k[:2]}={w}|{k[3:]}':1 for k in list(d) if k[0] in 'LR'})
 return d

def i32(X:"csr_matrix")->"csr_matrix":
 """Cast the index arrays of the sparse matrix X to int32 in place and return X."""
 # scipy>=1.17 gives int64 sparse indices, which newer scikit-learn SGD rejects; values are unchanged
 X.indices=X.indices.astype('int32');X.indptr=X.indptr.astype('int32');return X
def predict(train:Iterable[Mapping[str,Any]],test:Sequence[Mapping[str,Any]],
            lex:Mapping[str,Mapping[str,Collection[str]]],counts:collections.defaultdict[str,collections.Counter[str]],
            arm:str,seed:int=42)->list[str]:
 """One predicted length pattern per token of test, chosen among the key's candidates in lex.

 arm 'frequency' is the prior: the candidate most frequent in training (counts), or for a key with no training count
 the one listed by most lexicon sources. Ties go to fewer long vowels, then to the larger pattern string. It reads
 neither train nor seed. Any other arm (see features) trains an SGDClassifier with log loss, 35 epochs at most and
 random_state seed, on the training tokens that have a vowel and two or more candidates, with whole patterns as
 classes. A test token gets the candidate of its vowel count with the highest predicted probability (0 for a
 pattern that is no class), ties going to the larger training count, then to the first candidate in lex.
 Every arm writes a word without candidates all short. test tokens are read through key, gold or gold_pattern (only
 for their vowel count) and the fields of features. Side effect: counts gains an empty Counter for each test key
 that has candidates but no training count."""
 if arm=='frequency':
  # Prior: most frequent in train; for a skeleton unseen in train, the pattern most lexicon sources attest; then fewest long
  return [max(lex.get(r['key'],{zeros(r['gold'])}),key=lambda p:(counts[r['key']][p],0 if counts[r['key']] else len(lex[r['key']][p]),-p.count('1'),p)) if lex.get(r['key']) else '0'*len(r['gold_pattern']) for r in test]
 from sklearn.feature_extraction import DictVectorizer
 from sklearn.linear_model import SGDClassifier
 # Train only on the lexicon-ambiguous training tokens with a vowel, then pick the most probable candidate per token
 eligible=[r for r in train if len(r['gold_pattern'])>0 and len(lex.get(r['key'],set()))>=2]
 vec=DictVectorizer();X=i32(vec.fit_transform([features(r,arm) for r in eligible])); y=[r['gold_pattern'] for r in eligible]
 model=SGDClassifier(loss='log_loss',max_iter=35,tol=1e-3,random_state=seed);model.fit(X,y)
 probs=model.predict_proba(i32(vec.transform([features(r,arm) for r in test])));classes=list(model.classes_)
 out=[]
 for r,ps in zip(test,probs):
  opts:Collection[str]=lex.get(r['key'],set()); fallback=zeros(r['gold'])
  opts=[p for p in opts if len(p)==len(fallback)]
  out.append(max(opts,key=lambda p:(ps[classes.index(p)] if p in classes else 0,counts[r['key']][p])) if opts else fallback)
 return out

def load_classes(path:str|Path)->dict[str,str]:
 """Read the ambiguity class CSV (columns skeleton and final) into a dict of skeleton -> final class (A, B or C)."""
 with open(path,encoding='utf8') as f:return {r['skeleton']:r['final'] for r in csv.DictReader(f)}

def evaluate(test:Sequence[Mapping[str,Any]],preds:Sequence[str|None],lex:Mapping[str,Collection[str]],
             counts:Mapping[str,Mapping[str,int]],cls:Mapping[str,str]|None=None)->dict[str,Any]:
 """Whole-token accuracy of preds (one pattern per token of test, in order), overall and per harder population.

 Returns {'n', 'accuracy'} plus a {'n', 'accuracy'} dict for each flag of flags() and, when cls (load_classes output)
 is non-empty, for class_A, class_B, class_C and class_unlisted. The class populations hold only lexicon-ambiguous
 tokens, and class_unlisted those whose key is not in cls. A population's accuracy is None when it is empty.
 test must not be empty."""
 out:dict[str,Any]={'n':len(test),'accuracy':sum(p==r['gold_pattern'] for r,p in zip(test,preds))/len(test)}
 for label in ['lexicon_ambiguous','attested_ambiguous','minority_reading']:
  ix=[i for i,r in enumerate(test) if flags(r,lex,counts)[label]]
  out[label]={'n':len(ix),'accuracy':sum(preds[i]==test[i]['gold_pattern'] for i in ix)/len(ix) if ix else None}
 # the ambiguity classes split the lexicon-ambiguous tokens; 'unlisted' are ambiguous skeletons outside the class file
 for c in (['A','B','C','unlisted'] if cls else []):
  ix=[i for i,r in enumerate(test) if len(lex.get(r['key'],()))>=2 and cls.get(r['key'],'unlisted')==c]
  out['class_'+c]={'n':len(ix),'accuracy':sum(preds[i]==test[i]['gold_pattern'] for i in ix)/len(ix) if ix else None}
 return out

def oracle(test:Iterable[Mapping[str,Any]],lex:Mapping[str,Collection[str]])->list[str]:
 """The achievable ceiling of any candidate chooser, as one pattern per token of test.

 It gives the gold when the gold is a candidate, else a wrong candidate (the smallest pattern string, so the result is
 deterministic). A word without any candidate gets all short, as every system writes it, which is right exactly when
 the gold is all short."""
 return [r['gold_pattern'] if r['gold_pattern'] in lex.get(r['key'],{}) else min(lex[r['key']]) if lex.get(r['key']) else '0'*len(r['gold_pattern']) for r in test]

def mcnemar(a:Sequence[str],b:Sequence[str],gold:Sequence[str])->dict[str,int|float]:
 """Exact paired McNemar test of two prediction lists a and b against gold.

 Counts the discordant tokens and gives the two-sided binomial p value of the smaller count at 0.5, or 1.0 when no
 token is discordant. Returns a_wrong_b_right, a_right_b_wrong, discordant and p_exact."""
 from scipy.stats import binomtest
 b01=sum(x!=g and y==g for x,y,g in zip(a,b,gold));b10=sum(x==g and y!=g for x,y,g in zip(a,b,gold));n=b01+b10
 return {'a_wrong_b_right':b01,'a_right_b_wrong':b10,'discordant':n,'p_exact':binomtest(min(b01,b10),n,0.5).pvalue if n else 1.0}

def agreement(a:Sequence[Mapping[str,Any]],b:Sequence[Mapping[str,Any]],
              lex:Mapping[str,Collection[str]])->dict[str,Any]:
 """Length agreement between two editions a and b (token dicts in text order) on the tokens that align.

 The two key sequences are aligned with difflib.SequenceMatcher (autojunk off) and the tokens of its equal blocks are
 paired. Returns aligned, a_total, b_total, alignment_rate_a (aligned / len(a)) and two dicts, 'all' for every pair
 and 'lexicon_ambiguous' for the pairs whose key has two or more candidates in lex, each with n, agreement (share of
 identical whole-token patterns) and cohen_kappa over the pattern strings. Both are None for an empty subset, and
 kappa also when one pattern is the only one in both editions. a must not be empty."""
 ak=[r['key'] for r in a];bk=[r['key'] for r in b]
 sm=difflib.SequenceMatcher(None,ak,bk,autojunk=False); pairs:list[tuple[Mapping[str,Any],Mapping[str,Any]]]=[]
 for op,i,j,k,l in sm.get_opcodes():
  if op=='equal':pairs.extend(zip(a[i:j],b[k:l]))
 from sklearn.metrics import cohen_kappa_score
 result:dict[str,Any]={'aligned':len(pairs),'a_total':len(a),'b_total':len(b),'alignment_rate_a':len(pairs)/len(a)}
 for label,subset in [('all',pairs),('lexicon_ambiguous',[(x,y) for x,y in pairs if len(lex.get(x['key'],set()))>=2])]:
  aa=[x['gold_pattern'] for x,y in subset];bb=[y['gold_pattern'] for x,y in subset]
  result[label]={'n':len(subset),'agreement':sum(x==y for x,y in zip(aa,bb))/len(aa) if aa else None,'cohen_kappa':float(cohen_kappa_score(aa,bb)) if aa and len(set(aa+bb))>1 else None}
 return result

def clean_tauber(s:str)->str:
 """Bring a Tauber form to the corpus convention of the macron as the only diacritic.

 Removes the dot above (ċ, ġ) and the ogonek, writes a circumflex as a macron and recomposes to NFC."""
 s=unicodedata.normalize('NFD',s).replace('\u0307','').replace('\u0328','').replace('\u0302','\u0304')
 return unicodedata.normalize('NFC',s).replace('ċ','c').replace('ġ','g')

def load_tauber(path:str|Path)->list[dict[str,Any]]:
 """Read Tauber's Beowulf gold into token dicts, as tokens() yields them.

 path: the pipe-separated file of Brunetti's tokenisation (brunetti-length.txt), 18 columns per row: cols[9] is the
 unmarked form, cols[10] the punctuation and cols[-1] the length-marked form, both forms passed through clean_tauber.
 A row may hold several words separated by spaces, and each becomes a token. A sentence ends at a row whose
 punctuation has '.', '!' or '?', and words after the last such row form a final sentence. Sentences get ids tauber:0,
 tauber:1 and so on, with source tauber_beowulf. Blank lines are skipped. Raises ValueError on a wrong column count,
 a marked/bare mismatch or unequal word counts.
 Prints the number of tokens and long marks."""
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

def per_word(train:Sequence[Mapping[str,Any]],test:Sequence[Mapping[str,Any]],pred:Mapping[str,Sequence[str]],
             lex:Mapping[str,Collection[str]],counts:Mapping[str,Mapping[str,int]],top:int=20,
             cls:Mapping[str,str]|None=None)->list[dict[str,str|int|float]]:
 """Per-word results for the top (default 20) most frequent lexicon-ambiguous keys of test, one row dict per key.

 pred: system name -> one pattern per token of test. A row has word (the key), n (its tokens in test), class (from
 cls, else unlisted), test_pattern_counts (JSON of its gold patterns), <system>_accuracy rounded to six decimals, and
 <work>_pattern_counts (JSON of its gold patterns in train) for each of the five TRAIN works, '{}' when the work has
 no token of the key, as for a work left out.
 Keys tied in frequency keep the order of their first token in test."""
 subset=[(i,r) for i,r in enumerate(test) if flags(r,lex,counts)['lexicon_ambiguous']]
 most=collections.Counter(r['key'] for _,r in subset).most_common(top)
 out=[]
 for word,n in most:
  indices=[i for i,r in subset if r['key']==word];gold=[test[i]['gold_pattern'] for i in indices]
  row={'word':word,'n':n,'class':(cls or {}).get(word,'unlisted'),'test_pattern_counts':json.dumps(dict(collections.Counter(gold)),ensure_ascii=False)}
  for name,values in pred.items():row[name+'_accuracy']=round(sum(values[i]==test[i]['gold_pattern'] for i in indices)/n,6)
  for source in ['bright_reader','matthew_ws','sweet_primer','physiologus','zupitza_grammar']:
   ct=collections.Counter(r['gold_pattern'] for r in train if r['source']==source and r['key']==word)
   row[source+'_pattern_counts']=json.dumps(dict(ct),ensure_ascii=False)
  out.append(row)
 return out

def main()->None:
 """Command line entry: run the prior, the oracle and the logistic models on each --tests set and save the results.

 --exclude-train is empty by default, so the CLI trains on all five works unless the report's zupitza_grammar (D3)
 is passed. --tauber defaults to data/brunetti-length.txt, not to a file in --data. Unknown --tests or
 --exclude-train names exit with status 2, and if <data>/ambiguity_classes.csv exists accuracy is also split by class.
 For each set it writes <set>_predictions.jsonl (each token without its sentence, with its flags and every system's
 prediction) and <set>_per_word.csv to --out. metrics.json gets per set every system's metrics, the mean and sample
 SD of each logistic arm over seeds 1 to 3, seed-1 McNemar tests and the per-word rows, plus a dataset block (sizes,
 lexicon, left-out works), the Harrison and Sharp vs Wyatt agreement, a note on the test and the seeds. An existing
 metrics.json is merged at the top level (this run's keys replace the old ones), and a run whose lexicon (origin,
 file name, SHA-256) or left-out works differ from it exits before writing any file. So does a run that asks for
 tauber_beowulf without the Tauber file. The merged metrics are also printed."""
 p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--out',default='results/experiment');p.add_argument('--tauber',default='data/brunetti-length.txt');p.add_argument('--lexicon',help='Optional externally sourced lexicon TSV/CSV');p.add_argument('--tests',default='soliloquies,tauber_beowulf',help='Comma list: john_ws (dev), soliloquies, tauber_beowulf (run one at a time to keep each run short)');p.add_argument('--exclude-train',default='',help='Comma list of training works to leave out, e.g. zupitza_grammar');a=p.parse_args()
 wanted=set(a.tests.split(','));exclude=[x for x in a.exclude_train.split(',') if x]
 if wanted-set(TESTS):p.error(f'unknown --tests {sorted(wanted-set(TESTS))}: use {",".join(TESTS)}')
 if set(exclude)-set(TRAIN):p.error(f'unknown --exclude-train {sorted(set(exclude)-set(TRAIN))}: use {",".join(TRAIN)}')
 cpath=Path(a.data)/'ambiguity_classes.csv'  # optional: without it no per-class accuracy is reported
 train,prose,hs,wy=build_data(a.data,exclude);lex,counts=lexicon(train,a.lexicon);cls=load_classes(cpath) if cpath.exists() else None;out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 summary:dict[str,Any]
 summary={'dataset':{'train':len(train),'test_prose':len(prose),'agreement_hs':len(hs),'agreement_wyatt':len(wy),'lexicon_origin':'training attested forms'+(' + external file' if a.lexicon else ' (proxy; NOT Clark Hall/Wiktionary/Bosworth-Toller)')}}
 if exclude:summary['dataset']['exclude_train']=exclude  # only when set: the default metrics.json stays as before
 if a.lexicon:summary['dataset'].update(lexicon_file=Path(a.lexicon).name,lexicon_sha256=hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest())
 mpath=out/'metrics.json'  # refuse a mismatched --out folder before any file in it is overwritten
 if mpath.exists() and any(json.loads(mpath.read_text(encoding='utf8')).get('dataset',{}).get(k)!=summary['dataset'].get(k) for k in ('lexicon_origin','lexicon_file','lexicon_sha256','exclude_train')):raise SystemExit(f'{mpath} was made with a different lexicon or training set; use another --out folder')
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
