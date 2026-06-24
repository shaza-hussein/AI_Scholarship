import time
import json
import random
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


COUNTRIES = {
    "Egypt": "55", 
    "Syria": "97", 
    "Jordan": "89", 
    "Saudi Arabia": "95"
}

STATUSES = {
    "Undergraduates": "1", 
    "Graduates": "3", 
    "Doctoral candidates/PhD students": "4"
}

BASE_URL = "https://www2.daad.de"

def setup_driver():
    """Initializes the browser to run efficiently in the background."""
    print(">>> Initializing the browser in Headless Mode (Low Resource Mode)...")
    options = Options()
    options.add_argument('--headless') 
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    
    # Disable image loading to save bandwidth and speed up browsing
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def get_scholarship_links(driver):
    """Phase 1: Fetch ALL scholarship detail links across multiple pages (Pagination)."""
    print(">>> Phase 1: Collecting ALL links from DAAD...")
    
    all_links = []
    page_number = 1
    
    while True:
        
        paginated_url = f"https://www2.daad.de/deutschland/stipendium/datenbank/en/21148-scholarship-database/?status=&origin=&subjectGrps=&daad=&intention=&q=&page={page_number}&lang=en"
        
        print(f" -> Scraping Page {page_number}...")
        driver.get(paginated_url)
        time.sleep(5) 
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        results = soup.select('ul.resultlist li.entry h2 a')
        
       
        if not results:
            print(f"Reached the end of the database. Last page with results was {page_number - 1}.")
            break
            
        page_links = []
        for a in results:
            href = a.get('href')
            if href:
                page_links.append(BASE_URL + href)
                
        print(f"    Found {len(page_links)} scholarships on page {page_number}.")
        all_links.extend(page_links)
        
        page_number += 1
            
    
    unique_links = list(set(all_links))
    print(f"\nSuccessfully collected a total of {len(unique_links)} unique scholarships from all pages.")
    
    return unique_links

def clean_html_text(html_content):
    """
    Professional cleaning function: Merges deep links, removes DAAD-specific 
    system artifacts, and formats text for AI assistant readability.
    """
    if not html_content:
        return "N/A"
    
    element = BeautifulSoup(str(html_content), 'html.parser')
    
    # 1. Remove visible clutter
    junk_selectors = '.print-only, .invisible-print, .visible-de, .hidden-desktop, .info-icon-wrapper, script, style'
    for tag in element.select(junk_selectors):
        tag.decompose()
        
    # 2. Extract deep links for AI context
    for a in element.find_all('a'):
        href = a.get('href')
        if href:
            if href.startswith('/'):
                href = "https://www2.daad.de" + href
            link_text = a.get_text(strip=True)
            a.replace_with(f" {link_text} [{href}] ")
        
    # 3. Format bullet points
    for li in element.find_all('li'):
        li.insert(0, "• ")
        
    # 4. Format main headers
    for h in element.find_all(['h3', 'h2']):
        h.insert(0, "\n--- ")
        h.append(" ---\n")
        
    text = element.get_text(separator='\n', strip=True)
    
    # 5. Remove DAAD system placeholders
    text = re.sub(r'##[A-Z]+##', '', text)
    
    # 6. Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_dynamic_details(driver, url):
    """Phase 2: Scrape dynamic details for specific country and education level."""
    driver.get(url)
    time.sleep(4) 
    
    results = []
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    title_elem = soup.find('h2', class_='title')
    scholarship_name = title_elem.get_text(strip=True) if title_elem else "N/A"
    scholarship_name = scholarship_name.replace('• DAAD', '').strip()

    for country_name, country_code in COUNTRIES.items():
        for status_name, status_code in STATUSES.items():
            print(f"    -> Extracting data for: {country_name} | {status_name}")
            
            # JavaScript to trigger the selection change and submit
            js_script = f"""
            try {{
                document.getElementById('select-status-detail').value = '{country_code}';
                document.getElementById('select-country-detail').value = '{status_code}';
                document.getElementById('stipdb-submit-detail').click();
            }} catch(e) {{}}
            """
            
            try:
                driver.execute_script(js_script)
                time.sleep(3) 
                
                updated_soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                overview_div = updated_soup.find('div', id='ueberblick')
                scholarship_details = clean_html_text(overview_div)
                
                description = "N/A"
                if overview_div:
                    obj_h3 = overview_div.find('h3', string=lambda t: t and 'Objective' in t)
                    if obj_h3 and obj_h3.find_next_sibling('p'):
                        description = obj_h3.find_next_sibling('p').get_text(strip=True)
                    else:
                        description = scholarship_details[:300] + "..."

                req_div = updated_soup.find('div', id='voraussetzungen')
                eligibility = clean_html_text(req_div)
                
                proc_div = updated_soup.find('div', id='prozess')
                application_process = clean_html_text(proc_div)
                
                # Precise Deadline extraction logic
                deadline_text = "Not Specified"
                if proc_div:
                    deadline_h3 = proc_div.find('h3', string=lambda t: t and 'Application deadline' in t)
                    if deadline_h3:
                        parts = []
                        for sibling in deadline_h3.find_next_siblings():
                            if sibling.name in ['h2', 'h3']:
                                break
                            
                            txt = sibling.get_text(separator=' | ', strip=True)
                            txt = re.sub(r'(\s*\|\s*)+', ' | ', txt)
                            
                            if txt:
                                parts.append(txt)
                        if parts:
                            deadline_text = "\n".join(parts)

                # Eligibility check to filter out non-matching results
                is_eligible = True
                if "Not Eligible" in eligibility or "If your status and/or country is not included" in eligibility:
                    is_eligible = False

                if is_eligible:
                    data = {
                        "scholarship_name": f"{scholarship_name} ({country_name} - {status_name})",
                        "country": "Germany", 
                        # "target_country": country_name, 
                        "degree_level": status_name,
                        "funding_type": "Fully/Partially Funded (Check Details)",
                        "awards_available": "Varies",
                        "deadline": deadline_text, 
                        "tags": ["DAAD", country_name, status_name, "Germany"],
                        "description": description, 
                        "scholarship_details": scholarship_details,
                        "eligibility": eligibility,
                        "application_process": application_process,
                        "application_link": url
                    }
                    results.append(data)
                
            except Exception as e:
                print(f"Error extracting {country_name} - {status_name}: {e}")
            
            # Anti-blocking delay
            time.sleep(random.uniform(1.0, 2.0))
            
    return results

def main():
    driver = setup_driver()
    try:
        # Phase 1 will now get all links across all pages
        urls = get_scholarship_links(driver)
        all_data = []
        
        if not urls:
            print("No links found. Exiting.")
            return

        print(f"\n>>> Phase 2: Extracting dynamic details for ALL {len(urls)} scholarships...")
        
        for index, url in enumerate(urls, 1):
            print(f"\n[{index}/{len(urls)}] Processing Scholarship: {url}")
            data_list = extract_dynamic_details(driver, url)
            all_data.extend(data_list)
            
        print("\n>>> Saving data to file...")
        filename = 'daad_scholarships_data_full.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
            
        print(f"\nSuccess! Saved {len(all_data)} specific scholarship variations to '{filename}'")

    finally:
        print(">>> Closing browser and cleaning up...")
        driver.quit()

if __name__ == "__main__":
    main()