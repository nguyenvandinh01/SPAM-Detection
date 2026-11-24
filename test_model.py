import joblib

model = joblib.load('models/spam_classifier.joblib')

print('=== TEST SPAM.TXT ===')
with open('spam.txt', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        pred = model.predict([line.strip()])[0]
        proba = model.predict_proba([line.strip()])[0][1]
        label = 'SPAM' if pred == 1 else 'NOT SPAM'
        print(f'{i}. {label} ({proba:.2f}) - {line.strip()[:60]}')

print(f'\nTotal spam messages tested: {len(lines)}')
spam_count = sum(1 for line in lines if model.predict([line.strip()])[0] == 1)
print(f'Correctly detected SPAM: {spam_count}/{len(lines)}')

print('\n=== TEST HAM.TXT ===')
with open('ham.txt', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        pred = model.predict([line.strip()])[0]
        proba = model.predict_proba([line.strip()])[0][1]
        label = 'SPAM' if pred == 1 else 'NOT SPAM'
        print(f'{i}. {label} ({proba:.2f}) - {line.strip()[:60]}')

print(f'\nTotal ham messages tested: {len(lines)}')
ham_count = sum(1 for line in lines if model.predict([line.strip()])[0] == 0)
print(f'Correctly detected NOT SPAM: {ham_count}/{len(lines)}')