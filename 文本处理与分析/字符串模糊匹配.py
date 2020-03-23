import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

print(fuzz.ratio('我在看《亲密关系》',"我在看亲密关系")) #? 简单匹配——（88）
print(fuzz.partial_ratio('我在看《亲密关系》',"我在看亲密关系"))  #?  非完全匹配——86
print(fuzz.token_sort_ratio('我在看《亲密关系》','《亲密关系》我正在看'))  #? 忽略顺序匹配——94

#* 用来返回模糊匹配的字符串和相似度
sample=['我在看《亲密关系》','看《亲密关系》','亲密关系']
print(process.extract('我的亲密关系',sample,limit=2))  #? 返回sample里匹配高的前两名
print(process.extractOne('我的亲密关系',sample))  #? 返回匹配度最好的sample
print(process.extractBests('我的亲密关系',sample))  #? 按照匹配度排序返回