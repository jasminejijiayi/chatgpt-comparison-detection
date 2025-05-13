import nltk
import os
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 打印当前的 NLTK 数据路径
print("Current NLTK data paths:")
for path in nltk.data.path:
    print(f" - {path}")

# 创建一个自定义目录来存储 NLTK 数据
custom_nltk_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(custom_nltk_path, exist_ok=True)
print(f"\nCreated custom NLTK data path: {custom_nltk_path}")

# 将自定义路径添加到 NLTK 的搜索路径中
nltk.data.path.insert(0, custom_nltk_path)  # 将自定义路径放在列表的最前面
print("\nUpdated NLTK data paths:")
for path in nltk.data.path:
    print(f" - {path}")

# 下载 Punkt 数据包到自定义目录
print("\nDownloading Punkt tokenizer...")
nltk.download('punkt', download_dir=custom_nltk_path, quiet=False)

# 下载 punkt_tab 资源
print("\nDownloading punkt_tab resource...")
try:
    nltk.download('punkt_tab', download_dir=custom_nltk_path, quiet=False)
except Exception as e:
    print(f"Error downloading punkt_tab: {e}")

# 确认下载位置
tokenizers_dir = os.path.join(custom_nltk_path, 'tokenizers')
punkt_dir = os.path.join(tokenizers_dir, 'punkt')
if os.path.exists(punkt_dir):
    print(f"\nPunkt tokenizer successfully downloaded to: {punkt_dir}")
else:
    print("\nFailed to download Punkt tokenizer.")

# 尝试使用 Punkt 分词器
try:
    from nltk.tokenize import sent_tokenize
    test_text = "This is a test. This is another test."
    result = sent_tokenize(test_text)
    print(f"\nSentence tokenization test: {result}")
    print("Punkt tokenizer is working correctly!")
except Exception as e:
    print(f"\nError testing Punkt tokenizer: {e}")
    
# 如果仍然失败，尝试一个替代方法
if 'result' not in locals():
    print("\nTrying alternative method...")
    try:
        # 手动加载 punkt 模型
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        import pickle
        
        # 尝试查找 punkt 模型文件
        punkt_model_paths = []
        for path in nltk.data.path:
            model_path = os.path.join(path, 'tokenizers', 'punkt', 'english.pickle')
            if os.path.exists(model_path):
                punkt_model_paths.append(model_path)
        
        if punkt_model_paths:
            print(f"Found punkt model at: {punkt_model_paths[0]}")
            with open(punkt_model_paths[0], 'rb') as f:
                tokenizer = pickle.load(f)
            
            result = tokenizer.tokenize(test_text)
            print(f"\nAlternative tokenization test: {result}")
            print("Alternative method is working correctly!")
        else:
            print("Could not find punkt model file.")
    except Exception as e:
        print(f"\nError with alternative method: {e}")
