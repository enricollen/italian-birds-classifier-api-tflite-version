from flask import Flask, render_template, request, redirect, jsonify, url_for
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings
import tensorflow as tf
import numpy as np
import json
from PIL import Image

app = Flask(__name__)

CLASSES = ['accipiter_gentilis', 'accipiter_nisus', 'acrocephalus_arundinaceus', 'acrocephalus_melanopogon',
           'acrocephalus_palustris', 'acrocephalus_schoenobaenus', 'acrocephalus_scirpaceus', 'actitis_hypoleucos',
           'aegithalos_caudatus', 'aegolius_funereus', 'aegypius_monachus', 'alauda_arvensis', 'alcedo_atthis',
           'alectoris_barbara', 'alectoris_graeca', 'alectoris_rufa', 'anas_acuta', 'anas_clypeata', 'anas_crecca',
           'anas_discors', 'anas_formosa', 'anas_penelope', 'anas_platyrhynchos', 'anas_querquedula', 'anas_strepera',
           'anser_albifrons', 'anser_anser', 'anser_erythropus', 'anser_fabalis', 'anthus_campestris',
           'anthus_cervinus', 'anthus_pratensis', 'anthus_spinoletta', 'anthus_trivialis', 'apus_apus', 'apus_pallidus',
           'aquila_chrysaetos', 'aquila_clanga', 'aquila_pomarina', 'ardea_alba', 'ardea_cinerea', 'ardea_purpurea',
           'ardeola_ralloides', 'arenaria_interpres', 'asio_flammeus', 'asio_otus', 'athene_noctua', 'aythya_ferina',
           'aythya_fuligula', 'aythya_marila', 'aythya_nyroca', 'bombycilla_garrulus', 'botaurus_stellaris',
           'branta_bernicla', 'branta_leucopsis', 'branta_ruficollis', 'bubo_bubo', 'bubulcus_ibis',
           'bucanetes_githagineus', 'bucephala_clangula', 'burhinus_oedicnemus', 'buteo_buteo', 'buteo_lagopus',
           'buteo_rufinus', 'calandrella_brachydactyla', 'calcarius_lapponicus', 'calidris_alba', 'calidris_alpina',
           'calidris_canutus', 'calidris_ferruginea', 'calidris_minuta', 'calidris_temminckii', 'calonectris_diomedea',
           'caprimulgus_europaeus', 'carduelis_cannabina', 'carduelis_carduelis', 'carduelis_chloris',
           'carduelis_citrinella', 'carduelis_corsicana', 'carduelis_flammea', 'carduelis_spinus',
           'carpodacus_erythrinus', 'certhia_brachydactyla', 'certhia_familiaris', 'cettia_cetti',
           'charadrius_alexandrinus', 'charadrius_dubius', 'charadrius_hiaticula', 'charadrius_morinellus',
           'chlidonias_hybridus', 'chlidonias_leucopterus', 'chlidonias_niger', 'chroicocephalus_genei',
           'chroicocephalus_ridibundus', 'ciconia_ciconia', 'ciconia_nigra', 'cinclus_cinclus', 'circaetus_gallicus',
           'circus_aeruginosus', 'circus_cyaneus', 'circus_macrourus', 'circus_pygargus', 'cisticola_juncidis',
           'clamator_glandarius', 'clangula_hyemalis', 'coccothraustes_coccothraustes', 'coloeus_monedula',
           'columba_livia', 'columba_oenas', 'columba_palumbus', 'coracias_garrulus', 'corvus_corax', 'corvus_corone',
           'corvus_frugilegus', 'coturnix_coturnix', 'crex_crex', 'cuculus_canorus', 'cursorius_cursor',
           'cygnus_bewickii', 'cygnus_cygnus', 'cygnus_olor', 'delichon_urbicum', 'dendrocopos_leucotos',
           'dendrocopos_major', 'dendrocopos_medius', 'dendrocopos_minor', 'dryocopus_martius', 'egretta_garzetta',
           'emberiza_cia', 'emberiza_cirlus', 'emberiza_citrinella', 'emberiza_hortulana', 'emberiza_leucocephalos',
           'emberiza_melanocephala', 'emberiza_pusilla', 'emberiza_schoeniclus', 'eremophila_alpestris',
           'erithacus_rubecula', 'falco_biarmicus', 'falco_cherrug', 'falco_columbarius', 'falco_eleonorae',
           'falco_naumanni', 'falco_peregrinus', 'falco_subbuteo', 'falco_tinnunculus', 'falco_vespertinus',
           'ficedula_albicollis', 'ficedula_hypoleuca', 'ficedula_parva', 'ficedula_semitorquata',
           'francolinus_francolinus', 'fringilla_coelebs', 'fringilla_montifringilla', 'fulica_atra',
           'galerida_cristata', 'gallinago_gallinago', 'gallinago_media', 'gallinula_chloropus', 'garrulus_glandarius',
           'gavia_arctica', 'gavia_immer', 'gavia_stellata', 'gelochelidon_nilotica', 'glareola_pratincola',
           'glaucidium_passerinum', 'grus_grus', 'gypaetus_barbatus', 'gyps_fulvus', 'haematopus_ostralegus',
           'haliaeetus_albicilla', 'hieraaetus_fasciatus', 'hieraaetus_pennatus', 'himantopus_himantopus',
           'hippolais_icterina', 'hippolais_polyglotta', 'hirundo_daurica', 'hirundo_rustica',
           'histrionicus_histrionicus', 'ichthyaetus_audouinii', 'ichthyaetus_melanocephalus', 'ixobrychus_minutus',
           'jynx_torquilla', 'lagopus_mutus', 'lanius_collurio', 'lanius_excubitor', 'lanius_minor',
           'lanius_phoenicuroides', 'lanius_senator', 'larus_argentatus', 'larus_cachinnans', 'larus_canus',
           'larus_fuscus', 'larus_hyperboreus', 'larus_ichthyaetus', 'larus_marinus', 'larus_michahellis',
           'larus_minutus', 'limicola_falcinellus', 'limosa_lapponica', 'limosa_limosa', 'locustella_luscinioides',
           'locustella_naevia', 'loxia_curvirostra', 'lullula_arborea', 'luscinia_luscinia', 'luscinia_megarhynchos',
           'luscinia_svecica', 'lymnocryptes_minimus', 'macronectes_giganteus', 'marmaronetta_angustirostris',
           'melanitta_fusca', 'melanitta_nigra', 'melanocorypha_calandra', 'mergellus_albellus', 'mergus_merganser',
           'mergus_serrator', 'merops_apiaster', 'microcarbo_pygmeus', 'miliaria_calandra', 'milvus_migrans',
           'milvus_milvus', 'monticola_saxatilis', 'monticola_solitarius', 'montifringilla_nivalis', 'morus_bassanus',
           'motacilla_alba', 'motacilla_cinerea', 'motacilla_flava', 'muscicapa_striata', 'neophron_percnopterus',
           'netta_rufina', 'nucifraga_caryocatactes', 'numenius_arquata', 'numenius_phaeopus', 'nycticorax_nycticorax',
           'oenanthe_hispanica', 'oenanthe_isabellina', 'oenanthe_oenanthe', 'oriolus_oriolus', 'otis_tarda',
           'otus_scops', 'oxyura_jamaicensis', 'oxyura_leucocephala', 'pandion_haliaetus', 'panurus_biarmicus',
           'parus_ater', 'parus_caeruleus', 'parus_cristatus', 'parus_major', 'parus_montanus', 'parus_palustris',
           'passer_domesticus', 'passer_hispaniolensis', 'passer_italiae', 'passer_montanus', 'pelecanus_onocrotalus',
           'perdix_perdix', 'pernis_apivorus', 'petronia_petronia', 'phalacrocorax_aristotelis', 'phalacrocorax_carbo',
           'phalaropus_lobatus', 'phasianus_colchicus', 'philomachus_pugnax', 'phoenicopterus_roseus',
           'phoenicurus_ochruros', 'phoenicurus_phoenicurus', 'phylloscopus_bonelli', 'phylloscopus_collybita',
           'phylloscopus_inornatus', 'phylloscopus_proregulus', 'phylloscopus_sibilatrix', 'phylloscopus_trochilus',
           'pica_pica', 'picoides_tridactylus', 'picus_canus', 'picus_viridis', 'platalea_leucorodia',
           'plectrophenax_nivalis', 'plegadis_falcinellus', 'pluvialis_apricaria', 'pluvialis_squatarola',
           'podiceps_auritus', 'podiceps_cristatus', 'podiceps_grisegena', 'podiceps_nigricollis',
           'porphyrio_porphyrio', 'porzana_parva', 'porzana_porzana', 'prunella_collaris', 'prunella_modularis',
           'ptyonoprogne_rupestris', 'puffinus_yelkouan', 'pyrrhocorax_graculus', 'pyrrhocorax_pyrrhocorax',
           'pyrrhula_pyrrhula', 'rallus_aquaticus', 'recurvirostra_avosetta', 'regulus_ignicapillus', 'regulus_regulus',
           'remiz_pendulinus', 'riparia_riparia', 'rissa_tridactyla', 'saxicola_rubetra', 'saxicola_torquata',
           'scolopax_rusticola', 'serinus_serinus', 'sinosuthora_webbiana', 'sitta_europaea', 'somateria_mollissima',
           'somateria_spectabilis', 'stercorarius_longicaudus', 'stercorarius_parasiticus', 'stercorarius_pomarinus',
           'stercorarius_skua', 'sterna_caspia', 'sterna_hirundo', 'sternula_albifrons', 'streptopelia_decaocto',
           'streptopelia_turtur', 'strix_aluco', 'strix_uralensis', 'sturnus_roseus', 'sturnus_unicolor',
           'sturnus_vulgaris', 'sula_leucogaster', 'sylvia_atricapilla', 'sylvia_borin', 'sylvia_cantillans',
           'sylvia_communis', 'sylvia_conspicillata', 'sylvia_curruca', 'sylvia_hortensis', 'sylvia_melanocephala',
           'sylvia_nisoria', 'sylvia_rueppelli', 'sylvia_sarda', 'sylvia_undata', 'tachybaptus_ruficollis',
           'tachymarptis_melba', 'tadorna_ferruginea', 'tadorna_tadorna', 'tetrao_tetrix', 'tetrao_urogallus',
           'tetrastes_bonasia', 'tetrax_tetrax', 'thalasseus_sandvicensis', 'threskiornis_aethiopica',
           'tichodroma_muraria', 'tringa_erythropus', 'tringa_glareola', 'tringa_nebularia', 'tringa_ochropus',
           'tringa_stagnatilis', 'tringa_totanus', 'troglodytes_troglodytes', 'turdus_iliacus', 'turdus_merula',
           'turdus_philomelos', 'turdus_pilaris', 'turdus_torquatus', 'turdus_viscivorus', 'tyto_alba', 'upupa_epops',
           'vanellus_gregarius', 'vanellus_vanellus', 'xenus_cinereus']
CLASSES.sort()

# Initialize the predictor model
interpreter = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    img = request.files['image']

    ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'tiff'}
    if not allowed_file(img.filename, ALLOWED_EXTENSIONS):
        return redirect(url_for('result', species="not_able_to_predict"))

    img_preprocessed = image_preprocessing(img)
    predicted_species = compute_exact_prediction(img_preprocessed)
    compute_probability_distribution(img_preprocessed)

    if predicted_species == None:
        return redirect(url_for('result', species="not_able_to_predict"))
    return redirect(url_for('result', species=predicted_species))


@app.route('/result')
def result():
    species = request.args.get('species')

    if species == "not_able_to_predict":
        return render_template('error.html')

    species = species.replace("_", " ")
    species = species[0].upper() + species[1:]
    print("\nOne shot guess: the predicted species is " + species)

    return render_template('result.html', species=species)


def load_classifier():
    global interpreter

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="models/bird_classifier_no_optz.tflite")
    interpreter.allocate_tensors()

    return


def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions


def image_preprocessing(image_file):
    
    # Check file type
    file_extension = os.path.splitext(image_file.filename)[1]
    
    # Load and preprocess image based on file type
    if file_extension in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        # Load and preprocess JPEG image
        image = Image.open(image_file)
        img_preprocessed = image.resize((224, 224))
        img_preprocessed = np.array(img_preprocessed)
        img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_preprocessed)
    elif file_extension in (".png", ".PNG"):
        # Load and preprocess PNG image
        image = Image.open(image_file)
        img_preprocessed = image.convert('RGB')
        img_preprocessed = img_preprocessed.resize((224, 224))
        img_preprocessed = np.array(img_preprocessed)
        img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_preprocessed)
    elif file_extension in (".tiff", ".tif", ".TIFF", ".TIF"):
        # Load and preprocess TIFF image
        image = Image.open(image_file)
        img_preprocessed = image.resize((224, 224))
        img_preprocessed = np.array(img_preprocessed)
        img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_preprocessed)
    else:
        raise ValueError("Unsupported file type: {}".format(file_extension))
    
    return img_preprocessed


def compute_exact_prediction(img_preprocessed):
    global interpreter

    # Allocate memory for inputs
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], np.array([img_preprocessed], dtype=np.float32))

    # Run inference.
    interpreter.invoke()

    # Extract output results
    output_details = interpreter.get_output_details()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_tensor)
    label_name = CLASSES[predicted_class]

    return label_name


def compute_probability_distribution(img_preprocessed):
    global interpreter

    # Allocate memory for inputs
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], np.array([img_preprocessed], dtype=np.float32))

    # Run inference.
    interpreter.invoke()

    # Extract output results
    output_details = interpreter.get_output_details()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])

    # Prob Distribution of prediction
    probabilities = tf.nn.softmax(output_tensor, axis=-1)

    # Get the top k probabilities and indices
    k = 3
    top_k_values, top_k_indices = tf.math.top_k(probabilities, k=k)

    # Get the class labels associated with the top k indices
    top_k_classes = np.array(CLASSES)[tuple(top_k_indices)]

    # Print the top k classes and their associated probabilities
    for i in range(k):
        print("Class: {} | Probability: {}%".format(top_k_classes[i], top_k_values[0][i] * 100))
    
    return


# ----------------------------------------------- ONLY JSON REQUESTS BELOW ----------------------------------------------
def compute_probability_distribution_json(img_preprocessed, k=3):
    global interpreter, CLASSES
    
    # Allocate memory for inputs
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], np.array([img_preprocessed], dtype=np.float32))

    # Run inference
    interpreter.invoke()

    # Extract output results
    output_details = interpreter.get_output_details()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])

    # Compute probability distribution
    probabilities = tf.nn.softmax(output_tensor, axis=-1)

    # Get the top k probabilities and indices
    k = 3
    top_k_values, top_k_indices = tf.math.top_k(probabilities, k=k)

    # Get the class labels associated with the top k indices
    top_k_classes = np.array(CLASSES)[tuple(top_k_indices)]

    # Create a dictionary mapping class names to predicted probabilities
    class_prob_tuples = {top_k_classes[i]: '{:.2f}'.format(float(top_k_values[0][i]*100)) for i in range(k)}

    # Return the dictionary as a JSON-encoded string
    return class_prob_tuples

@app.route('/bird-prediction', methods=['POST'])
def predict_bird():
    # Get the file from the user request
    if 'file' not in request.files:
        return jsonify({'error': 'No file in request'}), 400  # Bad Request

    img_file = request.files['file']

    # Check if the file has a valid extension
    if img_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400  # Bad Request

    ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'tiff'}
    if not allowed_file(img_file.filename, ALLOWED_EXTENSIONS):
        return jsonify({'error': 'Invalid file extension'}), 400  # Bad Request

    # Apply logic to the image
    img_preprocessed = image_preprocessing(img_file)
    predicted_species = compute_exact_prediction(img_preprocessed)

    if predicted_species == None:
        return jsonify({'error': 'Prediction failed'}), 400  # Failed prediction

    # Return the response
    return jsonify({'predicted_species': predicted_species}), 200  # OK


@app.route('/bird-probabilities-prediction', methods=['POST'])
def predict_bird_probabilities():
    # Get the file from the user request
    if 'file' not in request.files:
        return jsonify({'error': 'No file in request'}), 400  # Bad Request

    img_file = request.files['file']

    # Check if the file has a valid extension
    if img_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400  # Bad Request

    ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'tiff'}
    if not allowed_file(img_file.filename, ALLOWED_EXTENSIONS):
        return jsonify({'error': 'Invalid file extension'}), 400  # Bad Request

    # Apply logic to the image
    img_preprocessed = image_preprocessing(img_file)
    predicted_probabilities_dict = compute_probability_distribution_json(img_preprocessed, 3)

    if predicted_probabilities_dict == None:
        return jsonify({'error': 'Prediction failed'}), 400  # Failed prediction

    # Return the response
    return jsonify({'predicted_species': predicted_probabilities_dict}), 200  # OK


if __name__ == '__main__':

    load_classifier()

    app.run(debug=False, host="0.0.0.0")
