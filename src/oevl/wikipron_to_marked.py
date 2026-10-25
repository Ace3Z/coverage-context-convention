"""Convert WikiPron Old English IPA to marked orthography (heuristic vowel alignment).

Reads a WikiPron TSV (a word, a tab, then its broad IPA with spaces between segments) and writes a CSV whose single
column, marked_form, lists each distinct spelling with macrons on the long vowels. run_all.sh builds
data/lex_wikipron.csv from data/ang_latn_broad.tsv this way. build_report.py calls mark() again to recompute the
alignment counts the paper prints, and charsiu_g2p.py and paper_numbers.py use mark() to read vowel length off
CharsiuG2P's predicted IPA.

Usage: python -m oevl.wikipron_to_marked IN.tsv OUT.csv
"""
import re,sys,unicodedata,collections,csv
# OV: orthographic vowel letters. IPA_V: IPA symbols that start a nucleus. DIPH: digraphs read as one vowel unit.
# MAC: the macron form of each vowel letter.
OV='aeiouyæ'; IPA_V=set('aeiouyæɑɒɛɪʊœøəɐʏɔ')
DIPH={'ea','eo','ie','io'}
MAC={'a':'ā','e':'ē','i':'ī','o':'ō','u':'ū','y':'ȳ','æ':'ǣ'}
def ipa_nuclei(ipa:str)->list[bool]:
 """Long-vowel flag of each vowel nucleus in a WikiPron IPA string, in order.

 Spaces (WikiPron's segment separators) are removed first. A nucleus starts at any symbol in IPA_V and takes in the
 diacritics that follow it (non-syllabic, syllabic, nasal tilde) and, across a tie bar, the next vowel, so a tied
 diphthong such as e͜oː is one nucleus. It is long when ː comes right after it. An untied non-syllabic vowel, as in
 æː ɑ̯, counts as a nucleus of its own. Every other symbol is skipped, including a precomposed vowel outside IPA_V
 such as ē (three WikiPron rows have one). Returns one bool per nucleus, True for long, or an empty list when there
 is no vowel."""
 s=ipa.replace(' ','');out=[];i=0
 while i<len(s):
  if s[i] in IPA_V:
   j=i+1
   while j<len(s) and (s[j] in '̯̩̃͜͡' or (s[j-1] in '͜͡' and s[j] in IPA_V)):j+=1
   long_=j<len(s) and s[j]=='ː';out.append(long_);i=j+(1 if long_ else 0)
  else:i+=1
 return out
def mark(bare:str,ipa:str):
 """Spell bare with a macron on each vowel unit whose IPA nucleus is long, or return None if the vowel counts differ.

 bare is lowercased and NFC-normalised. Its vowel units are the letters in OV, read left to right, with the digraphs
 ea, eo, ie and io taken as one unit, and the n-th unit is paired with the n-th nucleus from ipa_nuclei(ipa). A long
 unit gets the macron on its first letter (gōd, ēage, þēostrian). Other characters are copied unchanged, so the
 result is the lowercased word plus macrons. Only the counts are compared, not vowel quality, so a wrong pairing with
 equal counts goes unnoticed. bare should carry no macrons, as a letter that has one is not in OV. An empty ipa gives
 None for any word with a vowel letter. In WikiPron the counts differ mostly when ie is spoken as two vowels
 (stiriende, s t i r i e n d e) or i as j (berian, b e r j ɑ n)."""
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
 """Convert the WikiPron TSV at src into a marked-form CSV at out and return (stats, number of distinct forms).

 Each line is word, tab, IPA, and further columns are ignored. A line with fewer than two columns, a blank line
 included, raises ValueError before out is written. Rows whose lowercased word is not wholly a-z, þ, ð or æ count
 as skip_nonalpha, rows mark() cannot align count as skip_misaligned, and the rest as ok. stats is a Counter of these
 row counts. out is overwritten as UTF-8 with a header and the single column marked_form, holding the distinct marked
 forms in sorted order, so a word whose rows mark differently contributes each version. Prints the counts and the
 number of forms to stdout."""
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
