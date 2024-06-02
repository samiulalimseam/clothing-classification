pip install -r requirements.txt

DATA FLOW: 
API to get recommendations {Payload: product Id: String, customerId: String , domain: String} >>  returns {Recommended Products: Array of Product Id}

System: API will send required product and customer info. We will communicate with the client for product and customer data. Then our system will analyze the products that can be recommended . Our API will send a response with recommended product Id


