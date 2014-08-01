/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


package recognizer;
import com.googlecode.javacv.cpp.opencv_contrib.FaceRecognizer;
import static com.googlecode.javacv.cpp.opencv_contrib.createEigenFaceRecognizer;
import static com.googlecode.javacv.cpp.opencv_contrib.createFisherFaceRecognizer;
import static com.googlecode.javacv.cpp.opencv_contrib.createLBPHFaceRecognizer;
import com.googlecode.javacv.cpp.opencv_core;
import static com.googlecode.javacv.cpp.opencv_core.*;
import com.googlecode.javacv.cpp.opencv_core.CvFont;
import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_highgui;
import static com.googlecode.javacv.cpp.opencv_highgui.cvLoadImage;
import static com.googlecode.javacv.cpp.opencv_highgui.cvSaveImage;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_BGR2GRAY;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_INTER_LINEAR;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvCvtColor;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvEqualizeHist;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvResize;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvResize;
import static com.googlecode.javacv.cpp.opencv_objdetect.*;
import static com.googlecode.javacv.cpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;
import com.googlecode.javacv.cpp.opencv_objdetect.CvHaarClassifierCascade;
import static com.googlecode.javacv.cpp.opencv_objdetect.cvHaarDetectObjects;
import java.awt.Image;
import java.awt.Toolkit;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.*;

/**
 *
 * @author Owner
 */
public class FaceRecognition implements Runnable{
    
         // Create a new, second thread
      
    Main main = new Main();
    double accuracy = 10000;
    int SCALE = 2;
    String trainingDir = "images";
    File root = new File(trainingDir);
    boolean training;
    FilenameFilter pngFilter = new FilenameFilter() {
    public boolean accept(File dir, String name) {
         return name.toLowerCase().endsWith(".jpg");
         }
    };
    
   File[] imageFiles = root.listFiles(pngFilter);
   opencv_core.MatVector images = new opencv_core.MatVector(imageFiles.length);
   int[] labels = new int[imageFiles.length];
   int counter = 0;
   int label;
   opencv_core.IplImage img;
   opencv_core.IplImage grayImg;
   String[] names = new String[imageFiles.length];
    int width = 128;
    int height=150;     
    String[] labelName;
    int camID = 2;
    opencv_highgui.CvCapture capture = opencv_highgui.cvCreateCameraCapture(0);
    CvHaarClassifierCascade cascade =new CvHaarClassifierCascade(cvLoad("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"));
    opencv_core.IplImage grabbedImage = opencv_highgui.cvQueryFrame(capture);
    FaceRecognizer faceRecognizer = createFisherFaceRecognizer(0,accuracy);
    CvSeq faces;
    int total=0;
    IplImage original;
    CvRect r = new CvRect(0,0,0,0);
    int predict=0;
    Icon icon1,icon2,captured;
    int predicted=0;
    IplImage testImg1 = original;
    int nLabel,nFace;
    int name = 0;
    int prevLabel=0;
    boolean capturing = false;
    void getLabelFaceNumber(final String filename){
        FilenameFilter namesF = new FilenameFilter() {
            public boolean accept(File dir, String name) {
        return name.toLowerCase().contains(filename.toLowerCase());
            }
        };File[] labelFiles = root.listFiles(namesF);
        
        if(labelFiles.length>0){
        nLabel = Integer.parseInt(labelFiles[0].getName().split("\\_")[0]);

            int[] facenumber = new int[labelFiles.length];
            for(int i=0;i<facenumber.length;i++){
                String lastSplit;
                lastSplit = labelFiles[i].getName().split("\\_")[2];
                facenumber[i] = Integer.parseInt(lastSplit.substring(0,lastSplit.indexOf(".jpg")));
                nLabel = Integer.parseInt(labelFiles[i].getName().split("\\_")[0]);
            }
            Arrays.sort(facenumber);
            Arrays.sort(labels);
            nFace = facenumber[facenumber.length-1]+1;        
        }
        else{
           Arrays.sort(labels);
            nLabel = labels[labels.length-1]+1;
            nFace = 1;
        }
    }
    void readImgFiles(){
         root = new File(trainingDir);
     pngFilter = new FilenameFilter() {
   public boolean accept(File dir, String name) {
        return name.toLowerCase().endsWith(".jpg");
        }
   };
    
   imageFiles = root.listFiles(pngFilter);
   images = new opencv_core.MatVector(imageFiles.length);
   labels = new int[imageFiles.length];
   names = new String[imageFiles.length];
   counter = 0;
         for (File image : imageFiles) {
            img = cvLoadImage(image.getAbsolutePath());
            label = Integer.parseInt(image.getName().split("\\_")[0]);
            names[counter] =image.getName().split("\\_")[1];
            grayImg = opencv_core.IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);             
            cvCvtColor(img, grayImg, CV_BGR2GRAY);
            IplImage resizedImg = IplImage.create(128,150, IPL_DEPTH_8U, 1);
            cvResize(grayImg,resizedImg,CV_INTER_LINEAR);            
            IplImage testImg = IplImage.create(resizedImg.width(),resizedImg.height(), IPL_DEPTH_8U, 1);
            cvEqualizeHist(resizedImg, testImg);
            width = 128;
             height = 150;
            images.put(counter, testImg);
            labels[counter] = label;
            counter++;
        }
    }
    
    void getNames(){
        readImgFiles();
        Arrays.sort(labels);
        labelName =new String[labels[labels.length-1]];
        
        for(int i=0;i<labels.length;i++){
            if(i==0){
                labelName[i] = names[0];  
            }
            else if(labels[i]!=labels[i-1]){
                labelName[labels[i]-1] = names[i];
            }
        }      
    }
    
    void trainImg(){
       training = true;
       readImgFiles();       
       faceRecognizer.train(images, labels);
       getNames();
       System.out.println("train");
       
       training = false;
       
    }
    
    void captureImg2(){
       
        if(main.serialPort==null)
                main.initialize();
        
                grabbedImage = opencv_highgui.cvQueryFrame(capture);
                original = grabbedImage.clone();
                IplImage gray = IplImage.create(original.width(),original.height(), IPL_DEPTH_8U, 1);
                cvCvtColor(original, gray, CV_BGR2GRAY);
           
                IplImage smallImg = IplImage.create(gray.width()/SCALE,gray.height()/SCALE, IPL_DEPTH_8U, 1);
                cvResize(gray, smallImg, CV_INTER_LINEAR);

                // equalize the small grayscale
                IplImage equImg = IplImage.create(smallImg.width(),smallImg.height(), IPL_DEPTH_8U, 1);
                cvEqualizeHist(smallImg, equImg);

                // create temp storage, used during object detection
                CvMemStorage storage = CvMemStorage.create();

                // instantiate a classifier cascade for face detection

                 faces = cvHaarDetectObjects(equImg, cascade, storage,1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
                cvClearMemStorage(storage);
                
                IplImage cropped = original;               
                if(faces.total()>0) {
                    
                    r = new CvRect(cvGetSeqElem(faces, 0));
                    cvRectangle(original, cvPoint( r.x()*SCALE, r.y()*SCALE ),cvPoint( (r.x() + r.width())*SCALE,(r.y() + r.height())*SCALE ),CvScalar.RED, 6, CV_AA, 0);
                    recognize();
                    try {
                        if(main.serialPort!=null){
                            main.out((r.x()+r.width()/2));                                
                            main.out((r.y()+r.height()/2));                                
                        }                               
                    } catch (IOException ex) {
                                Logger.getLogger(FaceRecognition.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    
                }     
             
    }
    
    void recognize(){        
        if(faces==null) return;
                if(faces.total()>0) {
                        total =1;
                        prevLabel = predict;
                        predictImg(original); 
                }     
                
    }
    
    void predictImg(IplImage orig){
        if(r.x()>0&& r.width()>0&&r.height()>0 && orig!=null){
                  for(int i = 0 ; i<total; i++){      
                        CvRect R = new CvRect(r.x()*SCALE,r.y()*SCALE,r.width()*SCALE,r.height()*SCALE);
                        IplImage gray1 = IplImage.create(orig.width(),orig.height(), IPL_DEPTH_8U, 1);
                        cvCvtColor(orig, gray1, CV_BGR2GRAY); 
                        if(R.width()>0 && R.height()>0 && R.x()<orig.width() && r.y()<orig.height()){
                        cvSetImageROI(gray1, R);
                        IplImage cropped = cvCreateImage(cvGetSize(gray1), gray1.depth(), gray1.nChannels());
                        cvCopy(gray1, cropped);                          
                        
                        IplImage resizedImg = IplImage.create(width,height, IPL_DEPTH_8U, 1);
                        cvResize(cropped,resizedImg,CV_INTER_LINEAR);                 
                        
                        testImg1 = IplImage.create(resizedImg.width(),resizedImg.height(), IPL_DEPTH_8U, 1);
                        cvEqualizeHist(resizedImg, testImg1);
                        
                        int [] tabPredicted = {-1,-1};
                        double[] predConfTab = {0.5,0.5};
                       faceRecognizer.predict(testImg1,tabPredicted,predConfTab);
                       predicted = tabPredicted[0];
                       if(predicted>0){
                       String box_text = (names[tabPredicted[0]]);
                       
                       CvFont font = new CvFont(CV_FONT_HERSHEY_PLAIN, 2, 2); 
                       cvPutText(original, box_text, cvPoint(r.x()*SCALE-10, r.y()*SCALE-10), font,CV_RGB(0,255,0)); 
                       
                       this.predict = tabPredicted[0];
                      
                       }
                
        }
                if(total<1) predicted = 0;
        }}
    }
   
    void captureFace(String fileName, IplImage original){
        if(faces.total()>0){
            //training = "Updating Database.";
                        CvRect R = new CvRect(r.x()*SCALE,r.y()*SCALE,r.width()*SCALE,r.height()*SCALE);                        
                        IplImage gray1 = IplImage.create(original.width(),original.height(), IPL_DEPTH_8U, 1);
                        cvCvtColor(original, gray1, CV_BGR2GRAY); 
                        
                        cvSetImageROI(gray1, R);
                        IplImage cropped = cvCreateImage(cvGetSize(gray1), gray1.depth(), gray1.nChannels());
                        cvCopy(gray1, cropped);                          
                        
                        IplImage resizedImg = IplImage.create(width,height, IPL_DEPTH_8U, 1);
                        cvResize(cropped,resizedImg,CV_INTER_LINEAR);                
                        
                        IplImage testImg = IplImage.create(resizedImg.width(),resizedImg.height(), IPL_DEPTH_8U, 1);
                        cvEqualizeHist(resizedImg, testImg);
                        
                        getLabelFaceNumber(fileName);
                        
                        String filename = nLabel+"_"+fileName+"_"+nFace+".jpg";
                        System.out.println(filename);
                        cvSaveImage("images\\"+filename, testImg);
                        if(nFace==1){
                            filename = nLabel+"_"+fileName+"_"+nFace+++".jpg";
                            cvSaveImage("images\\"+filename, testImg);
                        }
                        readImgFiles();    
                        trainImg();
                        //}
                        
        }
    }
    
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
   

    @Override
    public void run() {
        trainImg();
        while(true){
            if(!training && !capturing)
                captureImg2();
                try {
                    Thread.sleep(100);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FaceRecognition.class.getName()).log(Level.SEVERE, null, ex);
                }
        }
    }



}
    

