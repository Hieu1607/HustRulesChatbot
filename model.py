import json
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# Cấu hình API key cho Gemini
genai.configure(api_key="AIzaSyDZij6g6vRtjZKBKABFngPHCXSP-na4EYY")  # Thay YOUR_API_KEY bằng key thật
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# Tải mô hình nhúng để so sánh ngữ nghĩa
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Hỗ trợ tiếng Việt

# Đọc file JSON
def load_rules(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Tìm kiếm câu hỏi tương đồng trong file JSON
def find_similar_question(user_question, rules, threshold=0.8):
    user_embedding = embedder.encode(user_question, convert_to_tensor=True)
    
    for rule in rules:
        rule_question = rule['question']
        rule_embedding = embedder.encode(rule_question, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(user_embedding, rule_embedding).item()
        
        if similarity >= threshold:
            return rule['answer']
    
    return None

# Trả lời câu hỏi bằng Gemini nếu không tìm thấy trong JSON
def answer_with_gemini(question, rules):
    # Tạo ngữ cảnh từ file JSON
    context = "\n".join([f"Hỏi: {rule['question']}\nTrả lời: {rule['answer']}" for rule in rules])
    
    prompt = f"""
    Dựa trên nội quy trường học sau đây, hãy trả lời câu hỏi một cách ngắn gọn và chính xác:
    {context}
    
    Câu hỏi: {question}
    """
    
    response = model.generate_content(prompt)
    return response.text

# Hàm chính để xử lý câu hỏi
def answer_school_rule(question, file_path='rules.json'):
    # Tải dữ liệu từ file JSON
    rules = load_rules(file_path)
    
    # Tìm câu trả lời trong file JSON
    answer = find_similar_question(question, rules)
    
    if answer:
        return answer
    else:
        # Nếu không tìm thấy, sử dụng Gemini
        gemini_answer = answer_with_gemini(question, rules)
        # (Tùy chọn) Lưu câu hỏi mới và câu trả lời vào file JSON
        save_new_rule(question, gemini_answer, file_path)
        return gemini_answer

# Lưu câu hỏi và câu trả lời mới vào file JSON
def save_new_rule(question, answer, file_path):
    rules = load_rules(file_path)
    rules.append({"question": question, "answer": answer})
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

# Ví dụ sử dụng
if __name__ == "__main__":
    file_path = 'rules.json'  # Đường dẫn đến file JSON
    questions = [
        "Học sinh có được mặc áo hoodie thay đồng phục không?",  # Câu hỏi có trong JSON
        "Nếu tôi đi trễ thì sao?",  # Câu hỏi tương tự nhưng không giống hệt
        "Có được ăn trong lớp không?"  # Câu hỏi không có trong JSON
    ]
    
    for q in questions:
        print(f"Câu hỏi: {q}")
        print(f"Trả lời: {answer_school_rule(q, file_path)}\n")