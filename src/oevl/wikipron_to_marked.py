"""Convert WikiPron Old English IPA to marked orthography (heuristic vowel alignment)."""
import re,sys,unicodedata,collections,csv
OV='aeiouyæ'; IPA_V=set('aeiouyæɑɒɛɪʊœøəɐʏɔ')
DIPH={'ea','eo','ie','io'}
MAC={'a':'ā','e':'ē','i':'ī','o':'ō','u':'ū','y':'ȳ','æ':'ǣ'}
def ipa_nuclei(ipa:str)->list[bool]:
 s=ipa.replace(' ','');out=[];i=0
 while i<len(s):
  if s[i] in IPA_V:
   j=i+1
   while j<len(s) and (s[j] in '̯̩̃͜͡' or (s[j-1] in '͜͡' and s[j] in IPA_V)):j+=1
   long_=j<len(s) and s[j]=='ː';out.append(long_);i=j+(1 if long_ else 0)
  else:i+=1
 return out
def mark(bare:str,ipa:str):
 w=unicodedata.normalize('NFC',bare.lower());units=[];i=0
 while i<len(w):
  if w[i] in OV:
   units.append(i);i+=2 if w[i:i+2] in DIPH else 1
  else:i+=1
 nuc=ipa_nuclei(ipa)
 if len(units)!=len(nuc):return None
 chars=list(w)
 for start,long_ in zip(units,nuc):
  if long_:chars[start]=MAC[chars[start]]
 return ''.join(chars)
def convert(src,out):
 stats=collections.Counter();forms=set()
 with open(src,encoding='utf-8') as f:
  for line in f:
   fields=line.rstrip('\n').split('\t')
   if len(fields)<2:raise ValueError('WikiPron TSV must have two columns')
   bare,ipa=fields[:2]
   if not re.fullmatch(r'[a-zþðæ]+',bare.lower()):stats['skip_nonalpha']+=1;continue
   m=mark(bare,ipa)
   if m is None:stats['skip_misaligned']+=1;continue
   stats['ok']+=1;forms.add(m)
 with open(out,'w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['marked_form']);w.writerows((x,) for x in sorted(forms))
 print(dict(stats),'distinct marked forms:',len(forms))
 return stats,len(forms)
if __name__=='__main__':
 if len(sys.argv)!=3:sys.exit('usage: python -m oevl.wikipron_to_marked IN.tsv OUT.csv')
 convert(sys.argv[1],sys.argv[2])
