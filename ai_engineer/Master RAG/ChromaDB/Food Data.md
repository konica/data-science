Part 1 Task 1: B. Download the Food Dataset
Download the comprehensive food dataset that will serve as the foundation for all three search systems:
wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/sN1PIR8qp1SJ6K7syv72qQ/FoodDataSet.json

This dataset contains rich food information including:

food_id: Unique identifier for each food item
food_name: Name of the dish
food_description: Detailed description of the food
food_calories_per_serving: Caloric content per serving
food_nutritional_factors: Breakdown of carbohydrates, protein, and fat
food_ingredients: List of ingredients used
food_health_benefits: Health benefits of the food
cooking_method: How the food is prepared (baking, grilling, etc.)
cuisine_type: Type of cuisine (American, Italian, etc.)
food_features: Taste, texture, appearance, and serving details
Example food item structure:

{
    "food_id": 1,
    "food_name": "Apple Pie",
    "food_description": "A classic dessert made with a buttery, flaky crust filled with tender, spiced apples.",
    "food_calories_per_serving": 320,
    "food_nutritional_factors": {
        "carbohydrates": "42g",
        "protein": "2g", 
        "fat": "16g"
    },
    "food_ingredients": ["Apples", "Flour", "Butter", "Sugar", "Cinnamon", "Nutmeg"],
    "food_health_benefits": "Rich in antioxidants and dietary fiber",
    "cooking_method": "Baking",
    "cuisine_type": "American",
    "food_features": {
        "taste": "sweet",
        "texture": "crisp and tender",
        "appearance": "golden brown",
        "preparation": "baked",
        "serving_type": "hot"
    }
}