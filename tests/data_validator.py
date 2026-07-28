import json
import re
import requests
import time

def validate_scholarship_data(json_file):
    print(f"Loading data from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found!")
        return

    print(f"Total Scholarships to verify: {len(data)}\n")
    
    # إحصائيات الجودة
    missing_deadlines = 0
    total_links_found = 0
    broken_links = []
    
    # إعداد جلسة requests للتحقق من الروابط
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    session.headers.update(headers)

    for index, item in enumerate(data, 1):
        # 1. التحقق من المواعيد النهائية المفقودة
        if item.get('deadline') == "Not Specified":
            missing_deadlines += 1
            
        # 2. استخراج جميع الروابط من هذه المنحة
        links_to_test = set()
        
        # الرابط الأساسي
        if item.get('application_link'):
            links_to_test.add(item['application_link'])
            
        # استخراج الروابط العميقة من النصوص باستخدام Regex (التي تكون بين أقواس مربعة)
        deep_links = re.findall(r'\[(https?://[^\s\]]+)\]', item.get('eligibility', '') + item.get('application_process', ''))
        for link in deep_links:
            links_to_test.add(link)
            
        total_links_found += len(links_to_test)
        
        # 3. فحص الروابط (أخذ عينة لتسريع الفحص، سنفحص أول منحتين كمثال)
        # ملاحظة: تم وضع الشرط أسفله لتجربة الفحص، يمكنك إزالة الشرط لفحص كل المنح
        if index <= 2: 
            print(f"Checking links for Scholarship [{index}]: {item['scholarship_name']}")
            for link in links_to_test:
                try:
                    # نستخدم HEAD بدلاً من GET لتسريع الفحص (يجلب حالة الرابط فقط دون تحميل الصفحة)
                    response = session.head(link, timeout=5, allow_redirects=True)
                    if response.status_code in [200, 201, 202, 301, 302]:
                        print(f"  [OK] {link}")
                    else:
                        print(f"  [BROKEN - Status {response.status_code}] {link}")
                        broken_links.append((item['scholarship_name'], link, response.status_code))
                except requests.RequestException as e:
                    print(f"  [FAILED - Network Error] {link}")
                    broken_links.append((item['scholarship_name'], link, "Network Error"))
            
            time.sleep(1) # لتجنب حظر السيرفر

    # طباعة تقرير الجودة النهائي
    print("\n" + "="*40)
    print("📊 DATA QUALITY REPORT")
    print("="*40)
    print(f"Total Scholarships Checked: {len(data)}")
    print(f"Scholarships missing precise deadlines: {missing_deadlines}")
    print(f"Total URLs extracted from texts: {total_links_found}")
    
    if broken_links:
        print(f"\n⚠️ Found {len(broken_links)} broken or unreachable links (in tested sample):")
        for name, link, error in broken_links:
            print(f" - {name}\n   Link: {link}\n   Error: {error}\n")
    else:
        print("\n✅ All tested links are working perfectly!")

if __name__ == "__main__":
    # ضع اسم ملفك هنا
    validate_scholarship_data('data_Json/computer_science_scholarships_data.json')