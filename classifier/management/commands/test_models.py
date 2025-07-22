from django.core.management.base import BaseCommand
from classifier.inference_service import PoliticalInferenceService

class Command(BaseCommand):
    help = 'Test the political classification models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--text',
            type=str,
            default="We need to protect democratic institutions from authoritarian threats.",
            help='Text to classify (default: demo text)'
        )
        parser.add_argument(
            '--context',
            type=str,
            default=None,
            help='Optional context for the text'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Testing political classification models...'))
        
        # Get inference service
        try:
            inference_service = PoliticalInferenceService.get_instance()
            self.stdout.write(self.style.SUCCESS('✓ Models loaded successfully'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Error loading models: {e}'))
            return

        # Test text
        text = options['text']
        context = options['context']
        
        self.stdout.write(f'\nAnalyzing text: "{text}"')
        if context:
            self.stdout.write(f'With context: "{context}"')

        # Run predictions
        try:
            results = inference_service.predict_all(text, context)
            coordinates = inference_service.get_3d_coordinates(text, context)
            
            self.stdout.write(self.style.SUCCESS('\n🎯 Classification Results:'))
            
            # Display results
            for model_name, result in results.items():
                if "error" in result:
                    self.stdout.write(self.style.WARNING(f'  ⚠️  {model_name}: {result["error"]}'))
                elif model_name == "manifesto_topics":
                    self.stdout.write(f'  📋 Policy Topic: {result["predicted_class"]}')
                else:
                    # Extract the score key dynamically
                    score_key = [k for k in result.keys() if k.startswith("predicted_")][0]
                    score = result[score_key]
                    interpretation = result["interpretation"]
                    dimension = model_name.replace("_regression", "").replace("_", "-").title()
                    self.stdout.write(f'  📊 {dimension}: {score} ({interpretation})')
            
            # Display 3D coordinates
            if coordinates['x'] is not None and coordinates['y'] is not None and coordinates['z'] is not None:
                self.stdout.write(self.style.SUCCESS('\n🎯 3D Coordinates:'))
                self.stdout.write(f'  X (Left-Right): {coordinates["x"]:.2f}')
                self.stdout.write(f'  Y (Liberal-Illiberal): {coordinates["y"]:.2f}')
                self.stdout.write(f'  Z (Populism): {coordinates["z"]:.2f}')
            
            # Display any errors
            if coordinates['errors']:
                self.stdout.write(self.style.WARNING('\n⚠️  Errors encountered:'))
                for error in coordinates['errors']:
                    self.stdout.write(f'  - {error}')
                    
            self.stdout.write(self.style.SUCCESS('\n✅ Model testing completed!'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Error during prediction: {e}'))
