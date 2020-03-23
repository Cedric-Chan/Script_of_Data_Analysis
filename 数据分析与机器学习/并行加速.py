'''
并行使用pandas（几十万行及以上才可以加速，否则反而不如直接运行）
'''
import modin.pandas as pd

cpl=pd.read_excel('desktop/CMPL.xlsx')  # 52s

#——————————————————————————————————————————————————————————————————————————————
'''
使用pandarallel并行加速库
'''
from pandarallel import pandarallel

#并行初始化
pandarallel.initialize(

start = time.time()   
emotion_df = weibo_df['review'].parallel_apply(emotion_caculate)
#emotion_df = weibo_df['review'].apply(emotion_caculate)
end = time.time()

print(end-start)
emotion_df.head()
#————————————————————————————————————————————————————————————————————
import pandas as pd

weibo_df = pd.read_csv('C:/Users/60448/Desktop/simplifyweibo_4_moods.csv')
weibo_df.head()