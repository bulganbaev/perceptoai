import os
import hailo_platform as hp


class HEFAnalyzer:
    def __init__(self, models_dir="data/models"):
        self.models_dir = models_dir
        self.model_path = None
        self.vdevice = hp.VDevice()
        self.network_group = None

    def list_models(self):
        """Отображает список доступных моделей в папке models_dir"""
        models = [f for f in os.listdir(self.models_dir) if f.endswith(".hef")]
        if not models:
            print("❌ Нет доступных HEF моделей в data/models")
            return []
        print("📌 Доступные модели:")
        for idx, model in enumerate(models, start=1):
            print(f"{idx}. {model}")
        return models

    def load_model(self):
        """Запрашивает у пользователя выбор модели и загружает её в виртуальное устройство"""
        models = self.list_models()
        if not models:
            return

        try:
            choice = int(input("🔹 Выберите номер модели: ")) - 1
            if choice < 0 or choice >= len(models):
                print("⚠️ Некорректный выбор! Попробуйте снова.")
                return
        except ValueError:
            print("⚠️ Введите корректное число!")
            return

        self.model_path = os.path.join(self.models_dir, models[choice])
        print(f"🚀 Загружаем модель: {models[choice]}")

        try:
            hef = hp.HEF(self.model_path)
            self.network_group = self.vdevice.configure(hef)[0]  # Берем первую (и единственную) сеть
        except Exception as e:
            print(f"❌ Ошибка при загрузке HEF: {e}")
            return

    def get_model_info(self):
        """Выводит информацию о входных и выходных слоях загруженной модели"""
        if not self.network_group:
            print("⚠️ Сначала загрузите модель!")
            return

        try:
            input_vstreams_info = self.network_group.get_input_vstream_infos()
            output_vstreams_info = self.network_group.get_output_vstream_infos()
        except Exception as e:
            print(f"❌ Ошибка при получении информации о потоках: {e}")
            return

        model_info = {
            "inputs": {},
            "outputs": {}
        }

        print("\n=== Входные потоки ===")
        for info in input_vstreams_info:
            model_info["inputs"][info.name] = {
                "shape": info.shape,
                "dtype": str(info.dtype)
            }
            print(f"📥 Name: {info.name}, Shape: {info.shape}, Data Type: {info.dtype}")

        print("\n=== Выходные потоки ===")
        for info in output_vstreams_info:
            model_info["outputs"][info.name] = {
                "shape": info.shape,
                "dtype": str(info.dtype)
            }
            print(f"📤 Name: {info.name}, Shape: {info.shape}, Data Type: {info.dtype}")

        return model_info


if __name__ == "__main__":
    analyzer = HEFAnalyzer()
    analyzer.load_model()
    config = analyzer.get_model_info()
