import json

SECTION_CORRECTIONS = {
    "Aloo Puri Recipe | Masala Potato Poori": ["breakfast"],
    "Besan Toast | Besan Bread Toast in Tawa - Just 10 Mins": ["breakfast"],
    "Choco Bar recipe | Chocobar Kids Favourite Ice Cream": ["sweets"],
    "cocktail samosa recipe | party samosa recipe with samosa sheets": ["snacks"],
    "Dal Fry Recipe - Dhaba Style with Secret tadka": ["dal"],
    "eggless chocolate cake recipe | eggless cake recipe": ["sweets"],
    "Kashmiri Pulao Recipe | Saffron Rice Recipe": ["rice"],
    "Masala Dosa Recipe | Crispy Masale Dose | How to make Masala Dosa": ["breakfast"],
    "Palkova Recipe | South Indian Special Palgova": ["sweets"],
    "Paneer Hyderabadi Masala | Hyderabadi Green Paneer Curry": ["paneer"],
    "Pani Puri Recipe - Street Style | Golgappa or Puchka - 5 Tips": ["snacks"],
    "poha vada recipe | aval vadai | flattened rice vada | aval masala vadi": ["breakfast"],
    "Rava Cake Recipe | Eggless Suji Cake": ["sweets"],
    "Burnt Garlic Fried Rice Recipe - Street Style": ["rice"],
}

with open("data/raw_recipes_clean.json", encoding="utf-8") as f:
    data = json.load(f)

fixed = 0
for r in data:
    if r["title"] in SECTION_CORRECTIONS:
        r["sections"] = SECTION_CORRECTIONS[r["title"]]
        fixed += 1

with open("data/raw_recipes_clean.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Corrected sections for {fixed} recipes")