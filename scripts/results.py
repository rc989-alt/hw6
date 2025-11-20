


from fcntl import F_EXLCK
import token


函数 evaluate（prefix):
    input prefix到模型 F_x  
    让模型自回归生成 T 个 token 
    统计整个序列中token 5的数量 
    返回该数量
    
初始化 prefix p = 随即长度 L 的 list（0-9） 
score = evaluate(p)

迭代n步