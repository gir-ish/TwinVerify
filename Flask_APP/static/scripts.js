let questions = [];
let embeddings = [[], [], []]; // Stores embedding objects per question
let recordingAttempts = [0, 0, 0]; // Tracks number of submissions per question
let mediaRecorders = [null, null, null]; // Stores MediaRecorder instances

document.getElementById('loadQuestions').addEventListener('click', () => {
    fetch('/get_questions')
        .then(response => response.json())
        .then(data => {
            questions = data.questions;
            document.getElementById('q1').innerText = questions[0];
            document.getElementById('q2').innerText = questions[1];
            document.getElementById('q3').innerText = questions[2];
            document.getElementById('questionsContainer').style.display = 'block';
            document.getElementById('loadQuestions').disabled = true;
            document.getElementById('reloadQuestions').disabled = false;
            // Enable the first question's Start Recording button
            document.getElementById('startBtn1').disabled = false;
        })
        .catch(error => {
            console.error("Error fetching questions:", error);
            alert("Failed to load questions. Please try again.");
        });
});

// document.getElementById('reloadQuestions').addEventListener('click', () => {
//     const username = document.getElementById('username').innerText;
//     // if (!confirm("Are you sure you want to reload questions? All previous recordings will be cleared.")) {
//     //     return;
//     // }
//     fetch('/get_questions')
//         .then(response => response.json())
//         .then(data => {
//             questions = data.questions;
//             document.getElementById('q1').innerText = questions[0];
//             document.getElementById('q2').innerText = questions[1];
//             document.getElementById('q3').innerText = questions[2];
//             resetAllQuestions();
//             alert("Questions reloaded. Previous recordings have been cleared.");
//             // Enable the first question's Start Recording button
            
//             document.getElementById('startBtn1').disabled = false;
//         })
//         .catch(error => {
//             console.error("Error reloading questions:", error);
//             alert("Failed to reload questions. Please try again.");
//         });
// });


// Reload Questions
document.getElementById('reloadQuestions').addEventListener('click', () => {
    fetch('/get_questions')
        .then(response => response.json())
        .then(data => {
            questions = data.questions;
            document.getElementById('q1').innerText = questions[0];
            document.getElementById('q2').innerText = questions[1];
            document.getElementById('q3').innerText = questions[2];
            
            // Re-enable the Start Recording buttons
            document.getElementById('startBtn1').disabled = false;
            // document.getElementById('startBtn2').disabled = false;
            // document.getElementById('startBtn3').disabled = false;

            alert("Questions reloaded successfully!");
        })
        .catch(error => {
            console.error("Error reloading questions:", error);
            alert("Failed to reload questions. Please try again.");
        });
});


document.getElementById('cancelEnrollment').addEventListener('click', () => {
    const username = document.getElementById('username').innerText;
    if (confirm("Are you sure you want to cancel the enrollment? All your data will be deleted.")) {
        const formData = new FormData();
        formData.append('user_id', username);
        fetch('/cancel_enrollment', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert("Enrollment canceled and data deleted.");
                window.location.href = "/";
            } else {
                alert("Failed to delete user data.");
            }
        })
        .catch(error => {
            console.error("Error deleting user data:", error);
            alert("Failed to delete user data.");
        });
    }
});

function startRecording(qIndex) {
    if (recordingAttempts[qIndex - 1] >= 3) {
        alert("Maximum recording attempts reached for this question.");
        return;
    }

    if (mediaRecorders[qIndex - 1]) {
        console.log(`Already recording for question ${qIndex}`);
        return;
    }

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorders[qIndex - 1] = mediaRecorder;
            let audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                mediaRecorders[qIndex - 1] = null;
                console.log(`Recording stopped for question ${qIndex}`);
                // Process the recorded audio
                processRecording(qIndex, audioBlob);
            };

            mediaRecorder.start();
            console.log(`Started recording for question ${qIndex}`);

            // Disable Start button, enable Stop button
            document.getElementById(`startBtn${qIndex}`).disabled = true;
            document.getElementById(`stopBtn${qIndex}`).disabled = false;

            // Stop all other recordings
            stopAllOtherRecordings(qIndex);
        })
        .catch(error => {
            console.error("Error accessing microphone:", error);
            alert("Could not access microphone. Please check your permissions.");
        });
}

function stopRecording(qIndex) {
    const mediaRecorder = mediaRecorders[qIndex - 1];
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        console.log(`Stopped recording for question ${qIndex}`);
    }
}

function processRecording(qIndex, audioBlob) {
    const username = document.getElementById('username').innerText;
    const question = questions[qIndex - 1];
    const formData = new FormData();
    formData.append('user_id', username);
    formData.append('question', question);
    formData.append('audio', audioBlob);

    // Show processing message
    document.getElementById(`transcript${qIndex}`).innerText = "Processing transcription...";
    document.getElementById(`error${qIndex}`).innerText = "";

    fetch('/submit_answer', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' || data.status === 'complete') {
            const transcription = data.transcription;
            const recording_id = data.recording_id;

            // Display transcription
            document.getElementById(`transcript${qIndex}`).innerText = "Transcription: " + transcription;

            // Enable Submit and Re-record buttons
            document.getElementById(`submitBtn${qIndex}`).disabled = false;
            document.getElementById(`reRecordBtn${qIndex}`).disabled = false;

            // Disable Stop button
            document.getElementById(`stopBtn${qIndex}`).disabled = true;

            // Increment attempt counter
            recordingAttempts[qIndex - 1] += 1;
            document.getElementById(`counter${qIndex}`).innerText = `Attempts: ${recordingAttempts[qIndex - 1]}/3`;

            // Add embedding to the list
            addEmbeddingToList(qIndex, recording_id, transcription, audioBlob);

            // Enable next question's Start Recording button if this was the first question
            if (qIndex < 3 && recordingAttempts[qIndex - 1] === 1) {
                document.getElementById(`startBtn${qIndex + 1}`).disabled = false;
            }

            // Check if all questions have at least one submitted recording
            checkAllAccepted();
        } else {
            // Display error message
            document.getElementById(`transcript${qIndex}`).innerText = "";
            document.getElementById(`error${qIndex}`).innerText = "Error: " + data.message;
            alert("Transcription failed. Please re-record your answer.");

            // Enable Re-record button
            document.getElementById(`reRecordBtn${qIndex}`).disabled = false;
        }
    })
    .catch(error => {
        console.error("Error submitting answer:", error);
        document.getElementById(`transcript${qIndex}`).innerText = "";
        document.getElementById(`error${qIndex}`).innerText = "Error submitting answer.";
        alert("Error submitting answer. Please try again.");

        // Enable Re-record button
        document.getElementById(`reRecordBtn${qIndex}`).disabled = false;
    });
}

function submitRecording(qIndex) {
    // Since the transcription is already processed and displayed, submitting will mark it as finalized
    alert("Your answer has been submitted successfully.");
    // Disable Submit and Re-record buttons to finalize the recording
    document.getElementById(`submitBtn${qIndex}`).disabled = true;
    document.getElementById(`reRecordBtn${qIndex}`).disabled = true;
    // Enable Delete button
    document.getElementById(`deleteBtn${qIndex}`).disabled = false;
    // Check if all questions have at least one submitted recording
    checkAllAccepted();
}

function reRecord(qIndex) {
    if (recordingAttempts[qIndex - 1] >= 3) {
        alert("Maximum recording attempts reached for this question.");
        return;
    }

    // Reset the recording and UI elements
    document.getElementById(`transcript${qIndex}`).innerText = "";
    document.getElementById(`error${qIndex}`).innerText = "";
    document.getElementById(`submitBtn${qIndex}`).disabled = true;
    document.getElementById(`reRecordBtn${qIndex}`).disabled = true;
    document.getElementById(`startBtn${qIndex}`).disabled = false;
}

function deleteRecording(qIndex, recording_id) {
    const username = document.getElementById('username').innerText;
    const question = questions[qIndex - 1];

    if (!confirm("Are you sure you want to delete this recording?")) {
        return;
    }

    const formData = new FormData();
    formData.append('user_id', username);
    formData.append('question', question);
    formData.append('recording_id', recording_id);

    fetch('/delete_recording', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert("Recording deleted successfully.");
            // Remove the embedding from the list
            const embeddingItem = document.getElementById(`embedding-${recording_id}`);
            if (embeddingItem) {
                embeddingItem.remove();
            }
            // Decrement attempt counter
            if (recordingAttempts[qIndex - 1] > 0) {
                recordingAttempts[qIndex - 1] -= 1;
                document.getElementById(`counter${qIndex}`).innerText = `Attempts: ${recordingAttempts[qIndex - 1]}/3`;
            }
            // Disable Finish button if necessary
            const finishBtn = document.getElementById('finishEnrollment');
            if (!isAllAnswered()) {
                finishBtn.disabled = true;
            }
        } else {
            alert("Failed to delete recording.");
        }
    })
    .catch(error => {
        console.error("Error deleting recording:", error);
        alert("Error deleting recording. Please try again.");
    });
}

function addEmbeddingToList(qIndex, recording_id, transcription, audioBlob) {
    const embeddingList = document.getElementById(`recordingList${qIndex}`);
    const embeddingItem = document.createElement('div');
    embeddingItem.className = 'recording-item';
    embeddingItem.id = `embedding-${recording_id}`;
    embeddingItem.innerHTML = `
        <p><strong>Transcription:</strong> ${transcription}</p>
        <button onclick="playEmbedding('${recording_id}')">Play</button>
        <button onclick="deleteRecording(${qIndex}, '${recording_id}')">Delete</button>
    `;
    embeddingList.appendChild(embeddingItem);
    // Store the embedding data for playback
    embeddings[qIndex - 1].push({ recording_id: recording_id, audioBlob: audioBlob });
}

function playEmbedding(recording_id) {
    // Find the audio blob from the embeddings array
    let audioBlob = null;
    for (let i = 0; i < embeddings.length; i++) {
        for (let j = 0; j < embeddings[i].length; j++) {
            if (embeddings[i][j].recording_id === recording_id) {
                audioBlob = embeddings[i][j].audioBlob;
                break;
            }
        }
        if (audioBlob) break;
    }

    if (audioBlob) {
        const audioURL = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioURL);
        audio.play();
    } else {
        alert("Recording not found.");
    }
}

function checkAllAccepted() {
    // Enable Finish Enrollment button if all questions have at least one submitted recording
    let allAnswered = true;
    for (let i = 0; i < 3; i++) {
        if (recordingAttempts[i] === 0) {
            allAnswered = false;
            break;
        }
    }
    document.getElementById('finishEnrollment').disabled = !allAnswered;
}

function finishEnrollment() {
    const user_id = document.getElementById('username').innerText;
    window.location.href = `/complete/${user_id}`;
}

function stopAllOtherRecordings(currentQIndex) {
    for (let i = 0; i < 3; i++) {
        if (i !== (currentQIndex - 1) && mediaRecorders[i]) {
            mediaRecorders[i].stop();
            mediaRecorders[i] = null;
        }
    }
}

function resetAllQuestions() {
    for (let i = 1; i <=3; i++) {
        embeddings[i-1] = [];
        recordingAttempts[i-1] = 0;
        document.getElementById(`transcript${i}`).innerText = "";
        document.getElementById(`error${i}`).innerText = "";
        document.getElementById(`counter${i}`).innerText = `Attempts: 0/3`;
        document.getElementById(`startBtn${i}`).disabled = true;
        document.getElementById(`stopBtn${i}`).disabled = true;
        document.getElementById(`submitBtn${i}`).disabled = true;
        document.getElementById(`reRecordBtn${i}`).disabled = true;
        document.getElementById(`deleteBtn${i}`).disabled = true;
        // Clear existing embeddings in the list
        const embeddingList = document.getElementById(`recordingList${i}`);
        embeddingList.innerHTML = '';
    }
    document.getElementById('finishEnrollment').disabled = true;
    // Enable Start Recording for the first question
    document.getElementById('startBtn1').disabled = false;
}

// Handle window unload to delete user data if enrollment not complete
window.addEventListener('beforeunload', function (e) {
    const user_id = document.getElementById('username').innerText;
    const allAnswered = recordingAttempts.every(count => count > 0);
    if (!allAnswered) {
        const formData = new FormData();
        formData.append('user_id', user_id);
        navigator.sendBeacon('/cancel_enrollment', formData);
    }
});
