"""Example usage of the Bayesian audiometry testing system without saved visualizations."""
from src.simulations.models.response_model import HearingResponseModel
from src.simulations.models.basic_bayes import BayesianPureToneAudiometry
def main():
    # Create simulated hearing profile
    hearing_profile = {
        250: 10,   
        500: 15,   
        1000: 20,  
        2000: 30,  
        4000: 45,  
        8000: 50   
    }
    # Configure response model
    response_params = {
        'slope': 10,      
        'guess_rate': 0.01,  
        'lapse_rate': 0.01,
        'threshold_probability': 0.5
    }
    # Initialize and run test
    audiometry = BayesianPureToneAudiometry(
        hearing_profile_data=hearing_profile,
        response_model_params=response_params,
        fp_rate=0.01,
        fn_rate=0.01,
        random_state=42
    )
    # Get results
    results = audiometry.perform_test()
if __name__ == "__main__":
    main()