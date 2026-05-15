import requests
from bs4 import BeautifulSoup
import time
import json
import random
import re  

# 1. Virtual browser settings to avoid blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

session = requests.Session()
session.headers.update(HEADERS)

def get_scholarship_links():
    """
    Phase 1: Collect links from the list page (table)
    """
    print(">>> Starting Phase 1: Collecting scholarship links...")
    scholarship_urls = []
    
    # URL for the scholarship list
    search_url = "https://www.scholarships.com/financial-aid/college-scholarships/scholarship-directory/academic-major/theology"
    
    print("Fetching scholarship list links...")
    try:
        response = session.get(search_url, timeout=10)
        
        if response.status_code != 200:
            print(f"Warning: Website refused connection! Status Code: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract links from the table
        link_elements = soup.select('td a.blacklink')
        
        if not link_elements:
            print("No links found in the table.")
            return []
            
        for a_tag in link_elements:
            href = a_tag.get('href')
            if href:
                full_link = f"https://www.scholarships.com{href}"
                if full_link not in scholarship_urls:
                    scholarship_urls.append(full_link)
        
        print(f"Successfully found {len(link_elements)} links.")
        
    except Exception as e:
        print(f"Error occurred while fetching the list: {e}")
            
    print(f"\nCollected {len(scholarship_urls)} links ready for Phase 2!\n")
    return scholarship_urls

def extract_scholarship_data(url):
    """
    Phase 2: Extract details with Tags, Awards Available, and Smart Text Matching for location/state
    """
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Extract Scholarship Name
        name_elem = soup.find('h1')
        scholarship_name = name_elem.text.strip() if name_elem else "N/A"
        
        # 2. Extract Deadline and Funding
        h5_tags = soup.find_all('h5')
        funding_type = "N/A"
        deadline = "N/A"
        
        if len(h5_tags) >= 2:
            funding_type = h5_tags[0].text.strip()
            deadline = h5_tags[1].text.strip()

        # ---------------------------------------------------------
        # 3. Extract Awards Available and Tags
        # ---------------------------------------------------------
        awards_available = "N/A"
        tags = []

        awards_span = soup.find('span', string=lambda text: text and 'Awards Available:' in text)
        if awards_span and awards_span.parent:
            awards_h5 = awards_span.parent.find_next_sibling('h5')
            if awards_h5:
                awards_available = awards_h5.text.strip()

        tags_heading = soup.find('h5', string=lambda text: text and ('Qualified Based On:' in text or 'Eligibility Criteria:' in text))
        if tags_heading:
            tags_container = tags_heading.find_next_sibling('div')
            if tags_container:
                tags = [tag.text.strip() for tag in tags_container.find_all('div')]

        # ---------------------------------------------------------
        # 4. Extract multiple text sections
        # ---------------------------------------------------------
        description = "N/A"
        details = "N/A"
        eligibility = "N/A"
        application_process = "N/A"
        
        main_desc_heading = soup.find('h2', string=lambda text: text and 'Scholarship Description' in text)
        
        if main_desc_heading:
            content_div = main_desc_heading.find_next_sibling('div')
            
            if content_div:
                first_p = content_div.find('p')
                if first_p:
                    description = first_p.get_text(strip=True)
                
                details_heading = content_div.find('h2', string=lambda text: text and 'Details' in text)
                if details_heading:
                    details_elem = details_heading.find_next_sibling()
                    if details_elem:
                        details = details_elem.get_text(separator="\n", strip=True)
                        
                eligibility_heading = content_div.find('h2', string=lambda text: text and 'Eligibility' in text)
                if eligibility_heading:
                    eligibility_elem = eligibility_heading.find_next_sibling()
                    if eligibility_elem:
                        eligibility = eligibility_elem.get_text(separator="\n", strip=True)
                        
                app_heading = content_div.find('h2', string=lambda text: text and 'Application Process' in text)
                if app_heading:
                    app_elem = app_heading.find_next_sibling()
                    if app_elem:
                        application_process = app_elem.get_text(separator="\n", strip=True)

        # ---------------------------------------------------------
        # 5. Smart Location Search (Regex method avoiding copyright)
        # ---------------------------------------------------------
        location = "USA"
        contact_div = soup.find('div', class_='contactDetail')
        
        if contact_div:
            lines = [l.strip() for line in contact_div.stripped_strings for l in line.split('\n') if l.strip()]
            for line in lines:
                # Pattern matching for US Address: City, ST 12345
                if re.search(r', [A-Z]{2} \d+', line):
                    location = f"{line} (USA)"
                    break
        
        if location == "USA":
            all_lines = [l.strip() for l in soup.get_text(separator="\n").split('\n') if l.strip()]
            for line in all_lines:
                if len(line) < 100 and re.search(r', [A-Z]{2} \d+', line):
                    location = f"{line} (USA)"
                    break
        # ---------------------------------------------------------
        
        # 6. Build enhanced data structure for JSON
        data = {
            "scholarship_name": scholarship_name,
            "country": location, 
            "degree_level": "Theology/Ministry", 
            "funding_type": funding_type,
            "awards_available": awards_available,
            "deadline": deadline,
            "tags": tags,
            "description": description,
            "scholarship_details": details,
            "eligibility": eligibility,
            "application_process": application_process,
            "application_link": url
        }
        return data

    except Exception as e:
        print(f"Failed to extract data from link {url}: {e}")
        return None

def main():
    urls = get_scholarship_links()
    all_scholarships_data = []
    
    if not urls:
        print("No links found to start Phase 2.")
        return

    print(">>> Starting Phase 2: Extracting scholarship details (this will take time to avoid blocking)...")
    
    for index, url in enumerate(urls, 1):
        print(f"Processing scholarship ({index}/{len(urls)})...")
        data = extract_scholarship_data(url)
        
        if data:
            all_scholarships_data.append(data)
            
        time.sleep(random.uniform(1.5, 3.0)) 
        
    print("\n>>> Saving data...")
    with open('theology_scholarships_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_scholarships_data, f, ensure_ascii=False, indent=4)
        
    print(f"Completed successfully! Saved {len(all_scholarships_data)} scholarships to 'theology_scholarships_data.json'")

if __name__ == "__main__":
    main()