// Hand Gesture Detection & Translation Scenarios & Data
// Matched with YouTube ISL Playlist Lessons (PLxYMaKXKMMcMgg4f47WkG7AM0bb3AyjTi)

export const ISL_CONVERSATIONS = [
  {
    id: 1,
    title: 'Greetings & Introduction (ISL Lesson 2)',
    exchanges: [
      {
        speaker: 'human',
        text: 'Hello! Namaste.',
        duration: 3000
      },
      {
        speaker: 'robot',
        text: 'Hello! Welcome to SignBridge.',
        duration: 3500
      },
      {
        speaker: 'human',
        text: 'How are you?',
        duration: 2500
      },
      {
        speaker: 'robot',
        text: 'I am doing well, thank you!',
        duration: 3000
      }
    ]
  },
  {
    id: 2,
    title: 'Casual Conversation (ISL Lesson 3)',
    exchanges: [
      {
        speaker: 'human',
        text: 'Nice to meet you.',
        duration: 2500
      },
      {
        speaker: 'robot',
        text: 'Nice to meet you too!',
        duration: 3000
      },
      {
        speaker: 'human',
        text: 'Thank you so much for your help.',
        duration: 3500
      },
      {
        speaker: 'robot',
        text: 'You are welcome! Happy to help anytime.',
        duration: 3500
      }
    ]
  },
  {
    id: 3,
    title: 'General Queries & Assistance',
    exchanges: [
      {
        speaker: 'human',
        text: 'Where is the washroom?',
        duration: 3000
      },
      {
        speaker: 'robot',
        text: 'The washroom is straight ahead to the left.',
        duration: 3500
      },
      {
        speaker: 'human',
        text: 'Can you please assist me?',
        duration: 3000
      },
      {
        speaker: 'robot',
        text: 'Yes, I am here to assist you.',
        duration: 3200
      }
    ]
  }
];

// Simple hand gesture detection outputs (User input pool)
export const HUMAN_INPUT_POOL = [
  'Hello! Namaste. 🙏',
  'How are you? 🤝',
  'Nice to meet you. 😊',
  'Thank you very much. 💖',
  'I need help. 🆘',
  'Where is the washroom? 🚻',
  'Good morning! 🌅',
  'Can you repeat that? 🔁',
  'Have a nice day! ✨',
  'Goodbye! 👋'
];

// Simple System/Assistant responses (Robot response pool)
export const ROBOT_INPUT_POOL = [
  'Hello! Welcome to SignBridge.',
  'I am doing well, thank you!',
  'Nice to meet you too!',
  'You are welcome!',
  'Yes, I am here to assist you.',
  'The washroom is straight ahead to the left.',
  'Good morning! How can I help you?',
  'Great! Let me know if you need anything.',
  'Have a wonderful day ahead!',
  'Goodbye! Take care.'
];
