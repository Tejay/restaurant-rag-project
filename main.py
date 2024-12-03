# main.py
from src.rag_system import RestaurantRAG

def main():
    rag_system = RestaurantRAG('data/')
    
    # Example queries
    query1 = "Best fine dining restaurants in London"
    query2 = "Top vegetarian restaurants with outdoor seating"
    
    print(rag_system.query(query1))
    print(rag_system.query(query2))

if __name__ == '__main__':
    main()