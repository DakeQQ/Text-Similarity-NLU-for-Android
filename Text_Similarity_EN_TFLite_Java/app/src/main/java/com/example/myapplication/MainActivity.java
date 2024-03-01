package com.example.myapplication;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.myapplication.databinding.ActivityMainBinding;

import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MainActivity extends AppCompatActivity {
    static final int font_size = 18;
    private static int amount_of_commands = 0;  // It will count automatically.
    private static final int cpu_threads = 4;
    private static final int top = 3;  // Assign a value to display the top K results.
    private static final int model_hidden_size_GTE = 768;
    private static final int token_unknown_GTE = 100;
    private static final int token_start_flag_GTE = 101;
    private static final int token_end_flag_GTE= 102;
    private static final int max_token_limit_GTE = 25;  // Replace with your max input size. This value is come from the model while exported.
    private static final float threshold_similar = 0.8f;  // You can print the max_score to assign a appropriate value.
    private static final int[] input_token_GTE = new int[max_token_limit_GTE];
    private static float[] score_pre_calculate;
    private static float[][] score_data_commands;
    private static String usrInputText = "";
    private static final String model_name = "Model_GTE.tflite";
    private static final String enter_question = "请输入问题 \nEnter Questions";
    private static final String cleared = "已清除 Cleared";
    private static final String not_found = "在列表中找不到相似指令。\nInsufficient similar commands found in the list.";
    private static final String file_name_vocab_GTE = "vocab_GTE.txt";
    private static final String file_name_commands = "commands.txt";  // You can edit this file, which is stored in the assets folder, to customize the personal commands.
    private static final List<String> vocab_GTE = new ArrayList<>();
    private static final List<String> commands = new ArrayList<>();
    Button sendButton;
    @SuppressLint("StaticFieldLeak")
    static AutoCompleteTextView inputBox;
    static RecyclerView answerView;
    private static ChatAdapter chatAdapter;
    private static List<ChatMessage> messages;
    private static InterpreterApi interpreter_A;
    private static ByteBuffer input_0;
    private static ByteBuffer input_1;
    private static final Map<Integer, Object> outputMap = new HashMap<>();;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        com.example.myapplication.databinding.ActivityMainBinding binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
        ImageView set_photo = findViewById(R.id.role_image);
        Button clearButton = findViewById(R.id.clear);
        sendButton = findViewById(R.id.send);
        inputBox = findViewById(R.id.input_text);
        messages = new ArrayList<>();
        chatAdapter = new ChatAdapter(messages);
        answerView = findViewById(R.id.result_text);
        answerView.setLayoutManager(new LinearLayoutManager(this));
        answerView.setAdapter(chatAdapter);
        set_photo.setImageResource(R.drawable.psyduck);
        clearButton.setOnClickListener(v -> clearHistory());
        AssetManager mgr = getAssets();
        if (Load_Models_0(mgr,false,false,false,false)) {
            Read_Assets(file_name_commands, mgr);
            Read_Assets(file_name_vocab_GTE, mgr);
            score_data_commands = new float[amount_of_commands][model_hidden_size_GTE];
            for (int i = 0; i < amount_of_commands; i++) {
                score_data_commands[i] = Run_Text_Embedding(Tokenizer(commands.get(i)));
                score_pre_calculate[i] = (float) Math.sqrt(Dot(score_data_commands[i], score_data_commands[i]));
            }
        } else {
            addHistory(ChatMessage.TYPE_SYSTEM, "模型加载失败。\nModel loading failed.");
        }
        Init_Chat();
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
    }

    private void Init_Chat() {
        sendButton.setOnClickListener(view -> {
            usrInputText = String.valueOf(inputBox.getText());
            inputBox.setText("");
            if (usrInputText.isEmpty()){
                showToast(enter_question,false);
                return;
            }
            addHistory(ChatMessage.TYPE_USER, usrInputText);
            long start_time = System.currentTimeMillis();
            int[] top_similar = Compare_Similarity(Tokenizer(usrInputText));
            addHistory(ChatMessage.TYPE_SERVER, "Time Cost: " + (System.currentTimeMillis() - start_time) + "ms\n\n");
            usrInputText = "";
            if (top_similar[0] != -1) {
                addHistory(ChatMessage.TYPE_SERVER,"前" + top + "名相似的指令如下:\n" + "The top " + top + " are as follows:\n");
                for (int i = 0; i < top; i++) {
                    addHistory(ChatMessage.TYPE_SERVER,"\nTop_" + (i + 1) + ": " + commands.get(top_similar[i]));
                }
            } else {
                addHistory(ChatMessage.TYPE_SERVER, not_found);
            }
        });
    }
    @SuppressLint("NotifyDataSetChanged")
    private static void addHistory(int messageType, String result) {
        int lastMessageIndex = messages.size() - 1;
        if (messageType == ChatMessage.TYPE_SYSTEM) {
            messages.add(new ChatMessage(messageType, result));
        } else if (lastMessageIndex >= 0 && messages.get(lastMessageIndex).type() == messageType) {
            if (messageType != ChatMessage.TYPE_USER ) {
                messages.set(lastMessageIndex, new ChatMessage(messageType, messages.get(lastMessageIndex).content() + result));
            } else {
                messages.set(lastMessageIndex, new ChatMessage(messageType, result));
            }
        } else {
            messages.add(new ChatMessage(messageType, result));
        }
        chatAdapter.notifyDataSetChanged();
        answerView.smoothScrollToPosition(messages.size() - 1);
    }
    @SuppressLint("NotifyDataSetChanged")
    private void clearHistory(){
        inputBox.setText("");
        messages.clear();
        chatAdapter.notifyDataSetChanged();
        answerView.smoothScrollToPosition(0);
        showToast(cleared,false);
    }
    private static int Search_Token_Index(@NonNull String word) {
        int index = vocab_GTE.indexOf(word);
        if (index != -1) {
            return index;
        }
        return token_unknown_GTE;
    }
    private static int Tokenizer(String question) {
        question = question.replaceAll("[-`~!@#$%^&*()_+=|{}':;\",\\[\\].<>/?·！￥…（）—《》【】‘；：”“’。，、？]","").toLowerCase(); // Remove the special sign.
        Matcher matcher = Pattern.compile("\\p{InCJK_UNIFIED_IDEOGRAPHS}|([a-zA-Z]+)|\\d|\\p{Punct}").matcher(question);
        input_token_GTE[0] = token_start_flag_GTE;
        int count = 1;
        while (matcher.find()) {
            String match = matcher.group();
            if (!match.trim().isEmpty()) {
                int search_result = Search_Token_Index(match);
                if (search_result != token_unknown_GTE) {
                    input_token_GTE[count] = search_result;
                    count++;
                    if (count == max_token_limit_GTE - 1) {
                        break;  // If the input query exceeds (max_token_limit_GTE - 2) words, it will be truncated directly.
                    }
                } else {
                    String[] match_split = match.split("");
                    if (match_split.length > 1) {
                        for (String words : match_split) {
                            input_token_GTE[count] = Search_Token_Index(words);
                            count++;
                            if (count == max_token_limit_GTE - 1) {
                                break;  // If the input query exceeds (max_token_limit_GTE - 2) words, it will be truncated directly.
                            }
                        }
                    } else {
                        input_token_GTE[count] = token_unknown_GTE;
                        count++;
                        if (count == max_token_limit_GTE - 1) {
                            break;  // If the input query exceeds (max_token_limit_GTE - 2) words, it will be truncated directly.
                        }
                    }
                }
            }
        }
        input_token_GTE[count] = token_end_flag_GTE;
        count++;
        return count;
    }
    private static int[] Find_TopK(float[] array) {
        int[] max_position = new int[top];
        for (int i = 1; i < top; i++) {
            max_position[i] = 0;
            float max_score = array[0];
            for (int j = 1; j < amount_of_commands; j++) {
                if (array[j] > max_score) {
                    max_score = array[j];
                    max_position[i] = j;
                }
            }
            array[max_position[i]] = -999.f;  // Assume -999 is the smallest value.
        }
        return max_position;
    }
    private static float Dot(float[] vector1, float[] vector2) {  // We have tried cblas_sdot, but it is 2 times slower.
        float sum = 0.f;
        for (int i = 0; i < model_hidden_size_GTE; i++) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    }
    private static int[] Compare_Similarity(int count) {
        float[] model_result = Run_Text_Embedding(count);
        float model_result_dot = (float) Math.sqrt(Dot(model_result, model_result));
        float[] temp_array = new float[amount_of_commands];
        temp_array[0] = Dot(score_data_commands[0], model_result) / (score_pre_calculate[0] * model_result_dot);
        float max_score = temp_array[0];
        int max_position = 0;
        for (int i = 1; i < amount_of_commands; i++) {
            temp_array[i] = Dot(score_data_commands[i], model_result) / (score_pre_calculate[i] * model_result_dot);
            if (temp_array[i] > max_score) {
                max_score = temp_array[i];
                max_position = i;
            }
        }
        temp_array[max_position] = -999.f;
        int[] TopK = Find_TopK(temp_array);
        if (max_score > threshold_similar) {
            TopK[0] = max_position;
        } else {
            TopK[0] = -1;
        }
        return TopK;
    }
    private void showToast(final String content, boolean display_long){
        if (display_long) {
            Toast.makeText(this, content, Toast.LENGTH_LONG).show();
        } else {
            Toast.makeText(this, content, Toast.LENGTH_SHORT).show();
        }
    }
    private void Read_Assets(String file_name, AssetManager mgr) {
        switch (file_name) {
            case file_name_vocab_GTE -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(mgr.open(file_name_vocab_GTE)));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        vocab_GTE.add(line);
                    }
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            case file_name_commands -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open(file_name_commands)));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        commands.add(line);
                        amount_of_commands += 1;
                    }
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    private boolean Load_Models_0(AssetManager mgr, boolean USE_XNNPACK, boolean USE_DSP_NPU, boolean USE_GPU, boolean USE_NNAPI) {
        InterpreterApi.Options options = new InterpreterApi.Options().setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_APPLICATION_ONLY);
        options.setUseXNNPACK(USE_XNNPACK);
        options.setNumThreads(cpu_threads);
        options.setCancellable(false);
        if (USE_DSP_NPU) {
            options.addDelegate(new HexagonDelegate(MainActivity.this));
        } else if (USE_GPU) {
            try (CompatibilityList compatList = new CompatibilityList()) {
                if (compatList.isDelegateSupportedOnThisDevice()) {
                    options.addDelegate(new GpuDelegate(compatList.getBestOptionsForThisDevice()));
                }
            } catch (Exception e) {
                return false;
            }
        } else if (USE_NNAPI) {
            options.setUseNNAPI(true);
            NnApiDelegate.Options nnapi_setting = new NnApiDelegate.Options();
            nnapi_setting.setAllowFp16(true);
            nnapi_setting.setUseNnapiCpu(false);
            nnapi_setting.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
            nnapi_setting.setMaxNumberOfDelegatedPartitions(8);
            options.addDelegate(new NnApiDelegate(nnapi_setting));
        }
        try (AssetFileDescriptor fileDescriptor = mgr.openFd(model_name);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            interpreter_A = InterpreterApi.create(inputStream.getChannel().map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength()), options);
        } catch (Exception e) {
            return false;
        }
        interpreter_A.allocateTensors();
        input_0 = ByteBuffer.allocateDirect(max_token_limit_GTE * 4).order(ByteOrder.nativeOrder());
        input_1 = ByteBuffer.allocateDirect(input_0.capacity()).order(ByteOrder.nativeOrder());
        outputMap.put(0, ByteBuffer.allocateDirect(model_hidden_size_GTE * 4).order(ByteOrder.nativeOrder()));
        return true;
    }
    private static float[] Run_Text_Embedding(int words_count) {
        for (int i = 0; i < words_count; i++) {
            input_0.putInt(input_token_GTE[i]);
            input_1.putFloat(1.f);
        }
        for (int i = words_count; i < max_token_limit_GTE; i++) {
            input_1.putFloat(-9999999999.f);
        }
        input_0.rewind();
        input_1.rewind();
        interpreter_A.runForMultipleInputsOutputs(new Object[] {input_0, input_1}, outputMap);
        ByteBuffer outputMap_byteBuffer = (ByteBuffer) outputMap.get(0);
        assert outputMap_byteBuffer != null;
        outputMap_byteBuffer.rewind();
        float[] floatArray = new float[model_hidden_size_GTE];
        outputMap_byteBuffer.asFloatBuffer().get(floatArray);
        return floatArray;
    }
}
