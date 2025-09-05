from alien_soil_env import AlienSoilEnv
from fix_engine import suggest_fixes

# Step 1: Create environment
env = AlienSoilEnv()

# Step 2: Reset and get a sample soil state
state = env.reset()
print("📊 Soil State:", state)

# Step 3: Take a random action (0 = not_habitable, 1 = moderate, 2 = habitat)
action = env.action_space.sample()
print("🎲 Action Chosen:", action)

# Step 4: Run environment step
next_state, reward, done, info = env.step(action)

# Step 5: Show result
print("🎯 Reward:", reward)
print("✅ Done:", done)
print("ℹ️ Info:", info)

# Step 6: If prediction is wrong (reward -1), call FixEngine
if reward == -1.0:
    soil_dict = dict(zip(env.features, state))
    print("\n🛠️ Suggested Fixes:")
    fixes = suggest_fixes(soil_dict)
    for k, v in fixes.items():
        print(f"{k}: {v}")
