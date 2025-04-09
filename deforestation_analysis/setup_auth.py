import ee

# Initialize Earth Engine
try:
    ee.Initialize()
    print("Earth Engine already initialized")
except Exception as e:
    # Get authentication URL
    ee.Authenticate()
    print("Authentication successful. Proceeding to initialize Earth Engine...")
    ee.Initialize(project='defanalysis')
    print("Earth Engine initialized successfully")