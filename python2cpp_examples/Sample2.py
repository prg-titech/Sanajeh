# 環境を見ながら新しい変数宣言を作る方法はいけない理由

if True:
    a1 = 4
    a2 = 5
    print(a1)
print(a1)
print(a2)
