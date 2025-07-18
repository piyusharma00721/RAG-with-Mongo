from pymongo import MongoClient

# Replace with your MongoDB Atlas connection string
client = MongoClient("mongodb+srv://sharmapiyush1106:N28xansYkZEb93Et@cluster0.ljxdkle.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client["langchain_test_db"]

try:
    db.drop_collection("local_rag")
    print("✅ Successfully dropped local_rag collection")
except Exception as e:
    print(f"❌ Error dropping collection: {e}")

# Verify
collections = db.list_collection_names()
print("Current collections:", collections)