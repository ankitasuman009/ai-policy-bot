import requests

from src.rag import protect_placeholders, restore_placeholders       
        
# url = "http://127.0.0.1:8000/chat/stream"

# payload = {
#     "question": "What does the handbook say about workplace harassment?"
# }

# with requests.post(url, json=payload, stream=True) as response:
#     response.raise_for_status()
#     for line in response.iter_lines(decode_unicode=True):
#         if line:
#             print(line)

sample = """
Employees receive [UNRESOLVED TEMPLATE VALUE: 20 days] of PTO.
The company offers [UNRESOLVED TEMPLATE VALUE: three months] of paid parental leave.
"""

protected_text, placeholder_map = protect_placeholders(sample)

print("--- PROTECTED TEXT ---")
print(protected_text)

print("\n--- RESTORED TEXT ---")
print(restore_placeholders(protected_text, placeholder_map))