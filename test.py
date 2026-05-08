import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def get_full_details(url, headers):
    """دالة للدخول إلى رابط المنحة وسحب النص الكامل للشروط والتفاصيل"""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            detail_soup = BeautifulSoup(response.content, 'html.parser')
            # البحث عن محتوى المقال (في هذا الموقع غالباً يكون داخل div بكلاس entry)
            entry_div = detail_soup.find('div', class_='entry')
            if entry_div:
                # تنظيف النص من الفراغات الزائدة
                return entry_div.get_text(separator=' ', strip=True)
        return "No details found"
    except Exception as e:
        return f"Error: {str(e)}"

def scrape_scholarships_project(max_pages=3):
    base_url = "https://www.scholars4dev.com/category/country/europe-scholarships/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_data = []

    for page in range(1, max_pages + 1):
        # التعامل مع روابط الصفحات
        if page == 1:
            current_url = base_url
        else:
            current_url = f"{base_url}page/{page}/"
        
        print(f"--- جاري العمل على الصفحة رقم {page} ---")
        
        try:
            response = requests.get(current_url, headers=headers)
            if response.status_code != 200:
                print(f"توقف الكود: الصفحة {page} غير موجودة.")
                break
                
            soup = BeautifulSoup(response.content, 'html.parser')
            posts = soup.find_all('div', class_='post')

            for post in posts:
                # 1. سحب المعلومات الأساسية من الفهرس
                title_tag = post.find('h2').find('a')
                title = title_tag.text.strip()
                link = title_tag['href']
                
                print(f"جاري سحب تفاصيل: {title}")
                
                # 2. الدخول لعمق المنحة لسحب (Eligibility, Deadline, Coverage)
                # ملاحظة: في هذه المرحلة نسحب النص الكامل والمساعد الذكي (AI) هو من سيصنفها لاحقاً
                full_details = get_full_details(link, headers)
                
                all_data.append({
                    "Title": title,
                    "Source_URL": link,
                    "Full_Content": full_details,
                    "Scraped_Date": time.strftime("%Y-%m-%d")
                })
                
                # انتظار عشوائي بين المنح لتجنب الحظر (بين 1 و 3 ثواني)
                time.sleep(random.uniform(1, 3))

            # انتظار بين الصفحات
            print(f"تم الانتهاء من الصفحة {page}. استراحة قصيرة...")
            time.sleep(5)

        except Exception as e:
            print(f"حدث خطأ في الصفحة {page}: {e}")
            continue

    return pd.DataFrame(all_data)

# --- تشغيل الكود ---
if __name__ == "__main__":
    # يمكنك تغيير رقم 2 إلى عدد الصفحات التي تريدها (مثلاً 10 أو 20)
    df_final = scrape_scholarships_project(max_pages=2)
    
    if not df_final.empty:
        # حفظ الملف بصيغة CSV تدعم العربية
        file_name = "./data/europe_scholarships_detailed.csv"
        df_final.to_csv(file_name, index=False, encoding='utf-8-sig')
        print(f"\n✅ تم بنجاح! إجمالي المنح المجموعة: {len(df_final)}")
        print(f"تم حفظ الملف باسم: {file_name}")
    else:
        print("❌ لم يتم جمع أي بيانات.")