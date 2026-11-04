# === Feature Engineering ==============================

def feature_construct(data, col='Message'):

    data = data.copy()                      
    text = data[col].astype(str)



    # --- length features ---
    data['char_len'] = text.str.len()                      
    data['word_count'] = text.str.split().str.len()  



    # --- number features ---
    data['digit_count'] = text.str.count(r'\d')            
    data['has_number'] = (data['digit_count'] > 0).astype(int) 
    data['number_count'] = text.str.count(r'\d+')
    data['has_phone'] = text.str.contains(r'\d{10,}').astype(int)



    # --- link feature ---
    data['has_url'] = text.str.contains(r'http|www\.|\.com|\.co\.uk', case=False).astype(int)



    # --- style features ---
    data['upper_ratio'] = text.str.count(r'[A-Z]') / data['char_len'].clip(lower=1)
    data['special_count'] = text.str.count(r'[^A-Za-z0-9\s]')
    data['exclaim_count'] = text.str.count('!')
    data['has_currency'] = text.str.contains(r'[£$€]').astype(int)



    # --- spam keyword feature ---
    keywords = r'\b(free|txt|stop|text|mobile|reply|claim|prize|cash|150p|urgent|won|nokia|service|msg|contact|win)\b'
    data['keyword_count'] = text.str.lower().str.count(keywords)



    # --- Droping Feature ---
    data.drop(columns=[col], inplace=True)



    return data