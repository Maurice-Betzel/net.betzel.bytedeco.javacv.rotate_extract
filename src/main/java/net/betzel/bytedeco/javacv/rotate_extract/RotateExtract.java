package net.betzel.bytedeco.javacv.rotate_extract;

import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

import static org.bytedeco.javacpp.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgproc.CHAIN_APPROX_NONE;

/**
 * Created by mbetzel on 08.06.2016.
 */
public class RotateExtract {

    static {
        System.setProperty("org.bytedeco.javacpp.logger", "slf4jlogger");
        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "debug");
    }

    private static final Slf4jLogger logger = (Slf4jLogger) org.bytedeco.javacpp.tools.Logger.create(RotateExtract.class);

    public static final void main(String[] args) {
        logger.info("Start");
        try {
            new RotateExtract().execute(args);
        } catch (Exception e) {
            e.printStackTrace();
        }
        logger.info("Stop");
    }

    private void execute(String[] args) throws Exception {
        // If no params provided, compute the default image
        BufferedImage bufferedImage = args.length >= 1 ? ImageIO.read(new File(args[0])) : ImageIO.read(this.getClass().getResourceAsStream("/images/left.jpg"));
        logger.info("Image type: " + bufferedImage.getType());
        // Convert BufferedImage to Mat and create AutoCloseable objects
        try (opencv_core.Mat matrix = new OpenCVFrameConverter.ToMat().convert(new Java2DFrameConverter().convert(bufferedImage));
             opencv_core.Mat bin = new opencv_core.Mat();
             opencv_core.Mat hierarchy = new opencv_core.Mat();
             opencv_core.MatVector contours = new opencv_core.MatVector()) {
            cvtColor(matrix, matrix, COLOR_BGR2GRAY);
            threshold(matrix, bin, 150, 255, THRESH_BINARY);
            findContours(bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
            logger.info("Countour count " + contours.size());
            for (int i = 0; i < contours.size(); ++i) {
                // Calculate the area of each contour
                opencv_core.Mat contour = contours.get(i);
                double area = contourArea(contour);
                // Ignore contours that are too small or too large
                if (area > 1024 && area < 1048576) {
                    rotateExtract(contour, matrix);
                }
            }
        }
    }

    private void rotateExtract(opencv_core.Mat contour, opencv_core.Mat matrix) {
        opencv_core.Mat rotatedMatrix = null;
        opencv_core.Size rotatedMatrixSize = null;
        opencv_core.Mat rotationMatrix = null;
        opencv_core.RotatedRect rotatedRect = null;
        try (opencv_core.Mat cropped = new opencv_core.Mat(); opencv_core.Size croppedSize = new opencv_core.Size()){
            rotatedRect = minAreaRect(contour);
            logger.info("Angle " + rotatedRect.angle());
            if (rotatedRect.size().width() < rotatedRect.size().height()) {
                rotatedRect.angle(rotatedRect.angle() + 90);
                float width = rotatedRect.size().width();
                float height = rotatedRect.size().height();
                rotatedRect.size().width(height);
                rotatedRect.size().height(width);
                rotatedMatrixSize = new opencv_core.Size(matrix.rows(), matrix.cols());
                rotatedMatrix = new opencv_core.Mat(rotatedMatrixSize, matrix.type());
            } else {
                rotatedMatrix = new opencv_core.Mat(matrix.size(), matrix.type());
            }
            int centerOffsetY = rotatedMatrix.rows() / 2 - (int) rotatedRect.center().y();
            int centerOffsetX = rotatedMatrix.cols() / 2 - (int) rotatedRect.center().x();
            rotationMatrix = getRotationMatrix2D(rotatedRect.center(), rotatedRect.angle(), 1.0);
            DoubleIndexer doubleIndexer = rotationMatrix.createIndexer();
            doubleIndexer.put(0, 2, doubleIndexer.get(0, 2) + centerOffsetX);
            rotatedRect.center().x(rotatedRect.center().x() + centerOffsetX);
            doubleIndexer.put(1, 2, doubleIndexer.get(1, 2) + centerOffsetY);
            rotatedRect.center().y(rotatedRect.center().y() + centerOffsetY);
            doubleIndexer.release();
            warpAffine(matrix, rotatedMatrix, rotationMatrix, rotatedMatrix.size(), INTER_CUBIC, BORDER_CONSTANT, opencv_core.Scalar.WHITE);
            showMatrix("WarpAffine", rotatedMatrix);
            croppedSize.height((int) rotatedRect.size().height());
            croppedSize.width((int) rotatedRect.size().width());
            getRectSubPix(rotatedMatrix, croppedSize, rotatedRect.center(), cropped);
            showMatrix("Rotated", cropped);
        } finally {
            if (rotationMatrix != null) {
                rotationMatrix.deallocate();
            }
            if (rotatedRect != null) {
                rotatedRect.deallocate();
            }
            if (rotatedMatrix != null) {
                rotatedMatrix.deallocate();
            }
            if (rotatedMatrixSize != null) {
                rotatedMatrixSize.deallocate();
            }
        }
    }

    protected static void showMatrix(String title, opencv_core.Mat matrix) {
        CanvasFrame canvasFrame = new CanvasFrame(title, 1);
        canvasFrame.setDefaultCloseOperation(javax.swing.JFrame.DISPOSE_ON_CLOSE);
        canvasFrame.setCanvasSize(matrix.size().width() / 4, matrix.size().height() / 4);
        Canvas canvas = canvasFrame.getCanvas();
        canvasFrame.getContentPane().removeAll();
        ScrollPane scrollPane = new ScrollPane();
        scrollPane.add(canvas);
        canvasFrame.add(scrollPane);
        canvasFrame.showImage(new OpenCVFrameConverter.ToMat().convert(matrix));
    }
}
