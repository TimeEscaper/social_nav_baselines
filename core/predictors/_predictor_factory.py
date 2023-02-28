from core.predictors import ConstantVelocityPredictor, NeuralPredictor, AbstractPredictor

class PredictorFactory:
    @staticmethod
    def create_predictor(config: dict) -> AbstractPredictor:
        if config["ped_predictor"] == "constant_velocity":
            predictor = ConstantVelocityPredictor(config["dt"],
                                                  config["total_peds"],
                                                  config["horizon"])

        elif config["ped_predictor"] == "neural":
                predictor = NeuralPredictor(total_peds=config["total_peds"],
                                            horizon=config["horizon"])
        
        return predictor