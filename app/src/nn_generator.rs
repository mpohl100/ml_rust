use evol::evolution::EvolutionOptions;
use evol::evolution::LogLevel;
use gen::neuralnet_gen::NeuralNetworkGenerator;
use neural::nn::shape::ActivationData;
use neural::nn::shape::ActivationType;
use neural::nn::shape::LayerShape;
use neural::nn::shape::LayerType;
use neural::nn::shape::NeuralNetworkShape;
use neural::training::data_importer::{DataImporter, SessionData};

use clap::Parser;
use neural::training::training_params::TrainingParams;
use neural::utilities::util::Utils;
use neural::utilities::util::WrappedUtils;

/// Command line arguments
#[derive(Parser)]
struct Args {
    /// Directory where the model shall be saved
    #[clap(long)]
    model_directory: String,
    #[clap(long, default_value = "")]
    shape_file: String,
    #[clap(long, default_value = "0")]
    retry_levels: i32,
    #[clap(long, default_value = "")]
    pre_shape: String,
    #[clap(long, default_value = "4")]
    num_threads: usize,
    #[clap(long, default_value = "1000000000")]
    cpu_memory: usize,

    // insert the training params here
    #[clap(long, default_value = "0.7")]
    validation_split: f64,
    #[clap(long, default_value = "0.01")]
    learning_rate: f64,
    #[clap(long, default_value = "100")]
    epochs: usize,
    #[clap(long, default_value = "0.1")]
    tolerance: f64,
    #[clap(long, default_value = "32")]
    batch_size: usize,
    #[clap(long, default_value = "false")]
    use_adam: bool,
    #[clap(long, default_value = "1.0")]
    sample_match_percentage: f64,

    // insert the evolution options here
    #[clap(long, default_value = "100")]
    num_generations: usize,
    #[clap(long, default_value = "1")]
    log_level: usize,
    #[clap(long, default_value = "4")]
    population_size: usize,
    #[clap(long, default_value = "10")]
    num_offsprings: usize,

    // insert data importer params here
    #[clap(long)]
    input_file: String,
    #[clap(long)]
    target_file: String,
}

impl Args {
    fn get_training_params(&self) -> TrainingParams {
        let levels = if self.retry_levels > 0 { Some(self.retry_levels) } else { None };
        let pre_shape = if self.pre_shape.is_empty() {
            None
        } else {
            Some(NeuralNetworkShape::from_file(self.pre_shape.clone()))
        };
        if self.shape_file.is_empty() {
            // deduce shape from input and target files
            let file_importer =
                FileDataImporter::new(self.input_file.clone(), self.target_file.clone());
            let data = file_importer.get_data();
            let input_size = data.data[0].len();
            let output_size = data.labels[0].len();
            // create a shape with one dense layer and the sigmoid activation function
            let shape = NeuralNetworkShape::new(vec![LayerShape {
                layer_type: LayerType::Dense { input_size, output_size },
                activation: ActivationData::new(ActivationType::Sigmoid),
            }]);
            return TrainingParams::new(
                shape,
                levels,
                pre_shape,
                self.validation_split,
                self.learning_rate,
                self.epochs,
                self.tolerance,
                self.batch_size,
                self.use_adam,
                self.sample_match_percentage,
            );
        }
        let shape = NeuralNetworkShape::from_file(self.shape_file.clone());
        // check dimensions of shape with input and target files
        let file_importer =
            FileDataImporter::new(self.input_file.clone(), self.target_file.clone());
        let data = file_importer.get_data();
        let input_size = data.data[0].len();
        let output_size = data.labels[0].len();
        assert_eq!(shape.layers[0].input_size(), input_size);
        assert_eq!(shape.layers[shape.layers.len() - 1].output_size(), output_size);
        TrainingParams::new(
            shape,
            levels,
            pre_shape,
            self.validation_split,
            self.learning_rate,
            self.epochs,
            self.tolerance,
            self.batch_size,
            self.use_adam,
            self.sample_match_percentage,
        )
    }

    fn get_evolution_options(&self) -> EvolutionOptions {
        EvolutionOptions::new(
            self.num_generations,
            match self.log_level {
                0 => LogLevel::None,
                1 => LogLevel::Minimal,
                2 => LogLevel::Verbose,
                _ => LogLevel::None,
            },
            self.population_size,
            self.num_offsprings,
        )
    }
}

// Mock DataImporter implementation for testing
#[derive(Clone)]
struct FileDataImporter {
    input_file: String,
    target_file: String,
}

impl FileDataImporter {
    fn new(
        input_file: String,
        target_file: String,
    ) -> Self {
        Self { input_file, target_file }
    }
}

impl DataImporter for FileDataImporter {
    fn get_data(&self) -> SessionData {
        // Initialize inputs and targets with zeros
        // read data from input csv file
        let data = self.read_data(self.input_file.clone());
        let labels = self.read_data(self.target_file.clone());

        SessionData { data, labels }
    }
}

impl FileDataImporter {
    fn read_data(
        &self,
        file: String,
    ) -> Vec<Vec<f64>> {
        let mut rdr = csv::Reader::from_path(file).unwrap();
        let mut data = Vec::new();
        for result in rdr.records() {
            let record = result.unwrap();
            let mut row = Vec::new();
            for value in record.iter() {
                row.push(value.parse::<f64>().unwrap());
            }
            data.push(row);
        }
        data
    }
}

fn main() {
    let args = Args::parse();
    let model_directory = &args.model_directory;
    let num_threads = args.num_threads;

    let training_params = args.get_training_params();

    let evolution_options = args.get_evolution_options();

    let data_importer = FileDataImporter::new(args.input_file, args.target_file);

    let utils = WrappedUtils::new(Utils::new(args.cpu_memory, args.num_threads));

    // generate from disk if the model_directory exists
    let mut nn_generator = if std::path::Path::new(model_directory).exists() {
        if args.shape_file.is_empty() {
            NeuralNetworkGenerator::from_disk(
                training_params,
                evolution_options,
                Box::new(data_importer),
                model_directory.clone(),
                num_threads,
                utils.clone(),
            )
        } else {
            NeuralNetworkGenerator::new(
                training_params,
                evolution_options,
                Box::new(data_importer),
                model_directory.clone(),
                num_threads,
                utils.clone(),
            )
        }
    } else {
        NeuralNetworkGenerator::new(
            training_params,
            evolution_options,
            Box::new(data_importer),
            model_directory.clone(),
            num_threads,
            utils.clone(),
        )
    };
    nn_generator.generate();
    nn_generator.save();
}
