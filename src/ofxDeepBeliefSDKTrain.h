#include "libjpcnn.h"
#include "ofMain.h"

class ofxDeepBeliefSDKTrain : public ofThread
{
  public:
    void* imageHandle;
    void* networkHandle;
    float* predictions;
    int predictionsLength;
    char** predictionsLabels;
    int predictionsLabelsLength;
    vector<string> vec;
    string pathnet;
    int doReverseChannels;
    int sourceRowBytes;
    int width, height;
    unsigned char* sourceStartAddr;
    void* trainer;
    void* predictor;
    int startPosNeg;

    void setup(string _pathnet){
        pathnet = _pathnet;
        startThread(true);
        ofLog()<<pathnet;
        trainer = NULL;
        predictor = NULL;
    }

    void exit(){
        stopThread();
    }

    void setPixels(unsigned char *u, int w, int h) {
        width = w;
        height = h;
        sourceStartAddr = u;
        doReverseChannels = 0;
        sourceRowBytes = (3 * width);
    }

    void update() {
        if(pathnet!="") {
            networkHandle = jpcnn_create_network(ofToDataPath(pathnet).c_str());
            void* cnnInput = jpcnn_create_image_buffer_from_uint8_data(sourceStartAddr, width, height, 4, sourceRowBytes, doReverseChannels, 1);
            jpcnn_classify_image(networkHandle, cnnInput, JPCNN_RANDOM_SAMPLE, -2, &predictions, &predictionsLength, &predictionsLabels, &predictionsLabelsLength);
            jpcnn_destroy_image_buffer(cnnInput);
            if(startPosNeg == 1)
            {
                jpcnn_train(trainer, 0.0f, predictions, predictionsLength);
            }
            if(startPosNeg == 2)
            {
                trainer = jpcnn_create_trainer();
                jpcnn_train(trainer, 1.0f, predictions, predictionsLength);
            }
            jpcnn_predict(predictor, predictions, predictionsLength);
        }
    }

    void startNegativeLearning()
    {
        startPosNeg = 1;
    }

    void startPositiveLearning()
    {
        if (trainer != NULL) {
           jpcnn_destroy_trainer(trainer);
        }
        if(predictor != NULL) {
           jpcnn_destroy_predictor(predictor);
        }
        startPosNeg = 2;
    }

    void stopLearning(){
        startPosNeg = 0;
    }

    void startPrediction(){
         if (predictor != NULL) {
           jpcnn_destroy_predictor(predictor);
         }
         predictor = jpcnn_create_predictor_from_trainer(trainer);
         //fprintf(stderr, "------------- SVM File output - copy lines below ------------\n");
         //jpcnn_print_predictor(predictor);
         //fprintf(stderr, "------------- end of SVM File output - copy lines above ------------\n");
    }

    void threadedFunction() {
        while(isThreadRunning()) {
            update();
        }
    }
};
