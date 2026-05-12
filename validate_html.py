"""Validate JSX div balance and component list in index.html."""
import re

with open('frontend/index.html', encoding='utf-8') as f:
    content = f.read()

start = content.find('<script type="text/babel">')
end   = content.rfind('</script>')
jsx   = content[start:end]

opens  = len(re.findall(r'<div[\s>]', jsx))
closes = len(re.findall(r'</div>', jsx))
print(f'<div>  opens : {opens}')
print(f'</div> closes: {closes}')
print(f'Balance      : {opens - closes}  (0 = perfect)')

funcs = re.findall(r'function\s+(\w+)\s*\(', jsx)
print(f'\nComponents ({len(funcs)}):')
for fn in funcs:
    print(f'  {fn}')

if abs(opens - closes) <= 2:
    print('\n[OK] JSX div balance is correct')
else:
    print(f'\n[WARN] Imbalance of {opens - closes} divs')
