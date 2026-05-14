import requests
from bs4 import BeautifulSoup
import time
import json
import random

# 1. Browser simulation settings to avoid blocking
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
    
    # URL for scholarship list (Theology/Academic Major as an example)
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
        print(f"An error occurred while fetching the list: {e}")
            
    print(f"\nCollected {len(scholarship_urls)} links ready for Phase 2!\n")
    return scholarship_urls

def extract_scholarship_data(url):
    """
    Phase 2: Access each link and extract details including state location
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
            
        # 3. Extract Description
        description = "N/A"
        desc_heading = soup.find('h2', string=lambda text: text and 'Scholarship Description' in text)
        if desc_heading:
            desc_div = desc_heading.find_next_sibling('div')
            if desc_div:
                description = desc_div.get_text(separator="\n", strip=True)

        # 4. Extract State from Contact section
        location = "USA" # Default value
        contact_div = soup.find('div', class_='contactDetail')
        
        if contact_div:
            contact_lines = contact_div.find_all('div', recursive=False)
            if len(contact_lines) >= 3:
                state_and_zip = contact_lines[2].text.strip()
                location = f"{state_and_zip} (USA)"
        
        # 5. Build JSON data structure
        data = {
            "scholarship_name": scholarship_name,
            "country": location, 
            "degree_level": "Theology/Ministry", 
            "funding_type": funding_type,
            "eligibility": description[:500] + "..." if description != "N/A" else "N/A",
            "deadline": deadline,
            "description": description,
            "application_link": url
        }
        return data

    except Exception as e:
        print(f"Failed to extract data from link {url}: {e}")
        return None

def main():
    # 1. Collect Links
    urls = get_scholarship_links()
    
    all_scholarships_data = []
    
    if not urls:
        print("No links found to start Phase 2. Aborting.")
        return

    # 2. Extract Details
    print(">>> Starting Phase 2: Extracting scholarship details (this will take time to avoid IP blocking)...")
    
    for index, url in enumerate(urls, 1):
        print(f"Processing scholarship ({index}/{len(urls)})...")
        data = extract_scholarship_data(url)
        
        if data:
            all_scholarships_data.append(data)
            
        # Time interval to avoid IP blocking (very important)
        time.sleep(random.uniform(1.5, 3.0)) 
        
    # 3. Save Final Data
    print("\n>>> Saving data...")
    with open('theology_scholarships_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_scholarships_data, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 Success! Saved {len(all_scholarships_data)} scholarships to 'theology_scholarships_data.json'")

if __name__ == "__main__":
    main()