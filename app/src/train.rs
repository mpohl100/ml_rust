use neural::nn::directory::Directory;
use neural::nn::shape::NeuralNetworkShape;
use neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};
use neural::training::data_importer::{DataImporter, SessionData};
use neural::training::training_params::TrainingParams;
use neural::training::training_session::TrainingSession;

use clap::Parser;
use neural::utilities::util::{Utils, WrappedUtils};

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
    let model_directory = args.model_directory.clone();

    let training_params = args.get_training_params();

    let data_importer = FileDataImporter::new(args.input_file, args.target_file);

    let utils = WrappedUtils::new(Utils::new(args.cpu_memory, args.num_threads));

    // if the model_directory exists load the model from disk
    let mut training_session = if std::fs::metadata(model_directory.clone()).is_ok() {
        if args.shape_file.is_empty() {
            TrainingSession::from_disk(
                model_directory.clone(),
                training_params,
                Box::new(data_importer),
                utils.clone(),
            )
            .expect("Failed to load model from disk")
        } else {
            TrainingSession::new(
                training_params,
                Box::new(data_importer),
                &Directory::User("internal_model".to_string()),
                utils.clone(),
            )
            .expect("Failed to create TrainingSession")
        }
    } else {
        TrainingSession::new(
            training_params,
            Box::new(data_importer),
            &Directory::Internal("internal_model".to_string()),
            utils.clone(),
        )
        .expect("Failed to create TrainingSession")
    };

    // Train the neural network and check the success rate
    let success_rate = training_session.train().expect("Training failed");
    // print the success rate
    println!("Success rate: {success_rate}");
    training_session.save_model(model_directory.clone()).expect("Failed to save model");
}
