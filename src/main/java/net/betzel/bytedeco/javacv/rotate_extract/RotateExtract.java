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
        BufferedImage bufferedImage = args.length >= 1 ? ImageIO.read(new File(args[0])) : ImageIO.read(this.getClass().getResourceAsStream("/images/right.jpg"));
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
        opencv_core.RotatedRect rotatedRect = minAreaRect(contour);
        float angle = rotatedRect.angle();
        logger.info("Angle " + angle);
        opencv_core.Size2f rect_size = rotatedRect.size();
        opencv_core.Mat rotated;
        if (rect_size.width() < rect_size.height()) {
            angle += 90.0;
            float width = rect_size.width();
            float height = rect_size.height();
            rect_size.width(height);
            rect_size.height(width);
            rotated = new opencv_core.Mat(new opencv_core.Size(matrix.rows(), matrix.cols()), matrix.type());
        } else {
            rotated = new opencv_core.Mat(matrix.size(), matrix.type());
        }
        int centerOffsetY = rotated.rows() / 2 - (int)rotatedRect.center().y();
        int centerOffsetX = rotated.cols() / 2 - (int)rotatedRect.center().x();
        opencv_core.Mat rotationMatrix = getRotationMatrix2D(rotatedRect.center(), angle, 1.0);
        DoubleIndexer doubleIndexer = rotationMatrix.createIndexer();
        doubleIndexer.put(0, 2, doubleIndexer.get(0, 2) + centerOffsetX);
        rotatedRect.center().x(rotatedRect.center().x() + centerOffsetX);
        doubleIndexer.put(1, 2, doubleIndexer.get(1, 2)  + centerOffsetY);
        rotatedRect.center().y(rotatedRect.center().y() + centerOffsetY);
        doubleIndexer.release();

        warpAffine(matrix, rotated, rotationMatrix, rotated.size(), INTER_CUBIC, BORDER_CONSTANT, opencv_core.Scalar.WHITE);
        showMatrix("WarpAffine", rotated);
        opencv_core.Mat cropped = new opencv_core.Mat();
        getRectSubPix(rotated, new opencv_core.Size((int) rect_size.width(), (int) rect_size.height()), rotatedRect.center(), cropped);
        showMatrix("Rotated", cropped);
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
