#include "libjpcnn.h"
#include "ofMain.h"

#define NETWORK_FILE_NAME "networks/jetpac.ntwk"

class ofxDeepBeliefSDK : public ofThread
{
	public:


		void load(string _pathimg, string _pathnet) {
			pathimg = _pathimg;
			pathnet = _pathnet;
			imageOrTexture = 0;
			startThread(true);
			ofLog()<<pathnet;
		}

		void setup(string _pathnet){
			pathnet = _pathnet;
			startThread(true);
			frame = false;;
			ofLog()<<pathnet;
		}

		void setPixels(unsigned char *u, int w, int h) {
			width = w;
			height = h;
	                sourceStartAddr = u;
			imageOrTexture = 1;
			doReverseChannels = 0;
			sourceRowBytes = (3 * width);
			frame = true;
		}

		void stop() {
			stopThread();
		}

                vector<string> getPrediction() {
        	    return vec;
	        }
		
		void printNetwork() {
		    jpcnn_print_network(networkHandle);
		}

	private:
		void* imageHandle;
	        void* networkHandle;
		float* predictions;
        	int predictionsLength;
	        char** predictionsLabels;
	        int predictionsLabelsLength;
		vector<string> vec;
		string pathimg,pathnet;
		int imageOrTexture;
		int doReverseChannels;
		int sourceRowBytes;
		int width, height;
		bool frame; 
		unsigned char* sourceStartAddr;

		void threadedFunction() 
		{
		     while(isThreadRunning()) {
			   if(imageOrTexture == 0) {
				if(pathimg!=""&&pathnet!="") {
					networkHandle = jpcnn_create_network(ofToDataPath(pathnet).c_str());
					imageHandle = jpcnn_create_image_buffer_from_file(ofToDataPath(pathimg).c_str());
					jpcnn_classify_image(networkHandle, imageHandle, 0, 0, &predictions, &predictionsLength, &predictionsLabels, &predictionsLabelsLength);
					int count = 0;
					for (int index = 0; index < predictionsLength; index += 1) {
						float predictionValue;
						char* label;
						predictionValue = predictions[index];
						if (predictionValue < 0.01f) {
							continue;
						}
						label = predictionsLabels[index];
						vec.push_back(ofToString(predictionValue)+" "+ofToString(label));
						count++;
					}
					if(vec.size() == count){
						ofLog()<<"Stop!";
						stopThread();
						jpcnn_destroy_image_buffer(imageHandle);
						jpcnn_destroy_network(networkHandle);
					}
			       }
			  }
			  else if(imageOrTexture == 1){
				  if(frame){
					  networkHandle = jpcnn_create_network(ofToDataPath(pathnet).c_str());
					  imageHandle = jpcnn_create_image_buffer_from_uint8_data(sourceStartAddr, width, height, 4, sourceRowBytes, doReverseChannels, 1);
//JPCNN_RANDOM_SAMPLE
					  jpcnn_classify_image(networkHandle, imageHandle, JPCNN_RANDOM_SAMPLE, 0, &predictions, &predictionsLength, &predictionsLabels, &predictionsLabelsLength);
                                          int count = 0;
					  vec.clear();
					  for (int index = 0; index < predictionsLength; index += 1) {
						    const float predictionValue = predictions[index];
						    if (predictionValue > 0.05f) {
						      char* label = predictionsLabels[index % predictionsLabelsLength];
				                      vec.push_back(ofToString(predictionValue)+" "+ofToString(label));
  						      count++;
						    }
					  }
                                          jpcnn_destroy_image_buffer(imageHandle);
                                          jpcnn_destroy_network(networkHandle);
				  }
			  }
		     }
		}

};
