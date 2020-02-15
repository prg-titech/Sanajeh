# 環境を見ながら新しい変数宣言を作る方法はいけない理由

b = {}
if True:
    a1 = 4
    a2 = 5

a = [1, 2, 3, 4]
for i in a[-2::-1]:
    print(i)
# setdefault(class_name.__name__, []).append(ob)
