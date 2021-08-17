package com.hashan.morsecode;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.os.Build;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import javax.xml.transform.Result;

public class MainActivity extends AppCompatActivity {

    Button button;
    EditText input;
    TextView mor;
    Interpreter tflite;
    private static final int SENTENCE_LEN = 256;
    private static final String SIMPLE_SPACE_OR_PUNCTUATION = " |\\,|\\.|\\!|\\?|\n";
    private final List<String> labels = new ArrayList<>();
    private final Map<String, Integer> dic = new HashMap<>();
    private static final String START = "<START>";
    private static final String PAD = "<PAD>";
    private static final String UNKNOWN = "<UNKNOWN>";

    /** Number of results to show in the UI. */
    private static final int MAX_RESULTS = 3;

    private static final String TAG = "Interpreter";

    private static final String MODEL_PATH = "text_classification.tflite";

    public static final String[] mAlphabet = {
            ". -",        // A
            "- . . .",    // B
            "- . - .",    // C
            "- . .",      // D
            ".",          // E
            ". . - .",    // F
            "- - .",      // G
            ". . . .",    // H
            ". .",        // I
            ".- - -",     // J
            "- . -",      // K
            ". - . .",    // L
            "- -",        // M
            "- .",        // N
            "- - -",      // O
            ". - - .",    // P
            "- - . -",    // Q
            ". - .",      // R
            ". . .",      // S
            "-",          // T
            ". . -",      // U
            ". . . -",    // V
            ". - -",      // W
            "- . . -",    // X
            "- . - -",    // Y
            "- - . ."     // Z
    };

    public static final String[] mNumbers = {
            ". - - - -",    // 0
            ". . - - -",    // 1
            ". . . - -",    // 2
            ". . . . -",    // 3
            ". . . . .",    // 4
            "- . . . .",    // 5
            "- - . . .",    // 6
            "- - - . .",    // 7
            "- - - - .",    // 8
            "- - - - -"     // 9
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        button = findViewById(R.id.button);
        input = findViewById(R.id.et);
        mor = findViewById(R.id.morse);

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception e){
            e.printStackTrace();
        }

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                String str_txt = input.getText().toString();
                String s = str_txt.toLowerCase();
                StringBuilder sb = new StringBuilder();

//                float prediction = inference(str_txt);
                int[][] input = tokenizeInputText(str_txt);
                String val = Integer.toString(input[1][0]);

                Toast.makeText(MainActivity.this, "this" +val, Toast.LENGTH_SHORT).show();

                for (char ch : s.toCharArray()) {
                    if (ch >= 'a' && ch <= 'z') {
                        int i = ch - 'a';

                        sb.append(mAlphabet[i]).append("   ");
                    }else if (ch >= '0' && ch <= '9') {
                        int j = ch - '0';
                        sb.append(mNumbers[j]).append("   ");
                    }
                }
                Vibrate(sb.toString().trim());
                mor.setText(sb.toString().trim());
            }
        });
    }

//    public float inference(String s){
//        float [] inputValue = new float[1];
//        inputValue[0] = Float.valueOf(s);
//
//        float[][] outputValue = new float[1][1];
//        tflite.run(inputValue,outputValue);
//        float inferredValue = outputValue[0][0];
//        return inferredValue;
//    }



    private void Vibrate(String temp) {

        Vibrator vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE);
        long dot = 100;
        long dash = 200;
        long short_gap = 500;

        ArrayList<Long> list = new ArrayList<Long>();
        list.add(new Long(0));
        for (char c : temp.toCharArray()) {
            if (c == '.') {
                list.add(dot);
            } else if (c == '-') {
                list.add(dash);
            } else if (c == ' ') {
                list.add(short_gap);
            }
        }
        long[] pattern = new long[list.size()];
        int i = 0;
        for (Long l : list) {
            long prim = l.longValue();
            pattern[i] = prim;
            i++;
        }
        vibrator.vibrate(pattern, -1);
    }

    private MappedByteBuffer loadModelFile() throws IOException{
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("activity-lite.tflite");
        FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffsets = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffsets,declaredLength);
    }

    /** Pre-prosessing: tokenize and map the input words into a float array. */
    int[][] tokenizeInputText(String text) {
        int[] tmp = new int[SENTENCE_LEN];
        List<String> array = Arrays.asList(text.split(SIMPLE_SPACE_OR_PUNCTUATION));

        int index = 0;
        // Prepend <START> if it is in vocabulary file.
        if (dic.containsKey(START)) {
            tmp[index++] = dic.get(START);
        }

        for (String word : array) {
            if (index >= SENTENCE_LEN) {
                break;
            }
            tmp[index++] = dic.containsKey(word) ? dic.get(word) : (int) dic.get(UNKNOWN);
        }
        // Padding and wrapping.
        Arrays.fill(tmp, index, SENTENCE_LEN - 1, (int) dic.get(PAD));
        int[][] ans = {tmp};
        return ans;
    }

    public synchronized List<Result> classify(String text) {
        // Pre-prosessing.
        String input = text;

        // Run inference.
        Log.v("", "Classifying text with TF Lite...");
        float[][] output = new float[1][labels.size()];
        tflite.run(input, output);

        // Find the best classifications.
//        PriorityQueue<Result> pq =
//                new PriorityQueue<>(
//                        MAX_RESULTS, (lhs, rhs) -> Float.compare(rhs.getConfidence(), lhs.getConfidence()));
//        for (int i = 0; i < labels.size(); i++) {
//            pq.add(new Result("" + i, labels.get(i), output[0][i]));
//        }
//        final ArrayList<Result> results = new ArrayList<>();
//        while (!pq.isEmpty()) {
//            results.add(pq.poll());
//        }
//
//        Collections.sort(results);
        // Return the probability of each class.
//        return results;
        return null;
    }

}