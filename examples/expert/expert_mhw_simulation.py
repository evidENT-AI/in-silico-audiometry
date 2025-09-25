"""Example usage of the expert audiometry testing system."""

from src.simulations.models.response_model import HearingResponseModel
from src.simulations.models.expert_mhw import ExpertModifiedHughsonWestlakeAudiometry
from src.visualization.simulation_plotting import (
    plot_audiogram,
    print_progression,
    plot_psychometric_comparison
)

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
    audiometry = ExpertModifiedHughsonWestlakeAudiometry(
        hearing_profile_data=hearing_profile,
        response_model_params=response_params,
        random_state=42
    )

    # Get results
    thresholds, progression_patterns = audiometry.perform_test()

    # Visualize results
    plot_audiogram(thresholds, "Example Audiogram")
    print_progression(progression_patterns[1000], 1000)
    plot_psychometric_comparison(
        HearingResponseModel,
        threshold_probabilities=[0.3, 0.5, 0.7]
    )

if __name__ == "__main__":
    main()