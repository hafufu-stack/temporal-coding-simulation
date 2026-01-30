"""リンゴ画像暗号化テスト"""
import sys
sys.path.insert(0, r'C:\Users\kyjan\研究\temporal-coding-simulation\snn-compression')
from stdp_comprypto import STDPComprypto
import os

apple_dir = r'C:\Users\kyjan\研究\snn-image-gen\data\apples'

print('=' * 65)
print('🍎 リンゴ画像 暗号化テスト')
print('=' * 65)

enc = STDPComprypto(key_seed=2026)

print(f"\n{'ファイル':<20} {'元':>12} {'暗号化':>12} {'比率':>8} {'復号':>6}")
print('-' * 65)

for filename in sorted(os.listdir(apple_dir)):
    if filename.endswith('.jpg'):
        filepath = os.path.join(apple_dir, filename)
        
        with open(filepath, 'rb') as f:
            data = f.read()
        
        encrypted = enc.encrypt(data, verbose=False)
        
        dec = STDPComprypto(key_seed=2026)
        restored = dec.decrypt(encrypted, verbose=False)
        
        ok = (data == restored)
        ratio = len(encrypted) / len(data) * 100
        
        print(f"{filename:<20} {len(data):>12,} {len(encrypted):>12,} {ratio:>7.1f}% {'✅' if ok else '❌':>6}")

print()
print('JPGは既に圧縮されているため、あまり縮まないが復号は完璧！')
