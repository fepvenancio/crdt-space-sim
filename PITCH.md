# PITCH.md - Cofounder & Investor Talking Points

## The 30-Second Pitch

> "We're building the coordination layer for autonomous space robot fleets. Using CRDT technology—the same tech behind Figma's multiplayer—our robots can work together without constant ground control. Our fair simulation shows CRDT coordination outperforms centralized control at Lunar distances and beyond—exactly where centralized fails. We're looking for a technical cofounder with robotics experience to take this from proof-of-concept to product."

---

## The Problem (2 minutes)

### For Technical Audiences

"Space robots today are basically remote-controlled toys. Every action needs a command from ground control. But there's a problem:

- **GEO satellites**: 0.5 second round-trip delay
- **Lunar operations**: 2.6 second delay
- **Mars**: 8-48 MINUTES each way

When you have 5 robots trying to service a satellite, they can't all wait for ground commands. The bandwidth isn't there, the latency kills efficiency, and during solar events or eclipse, you have no comms at all.

Current solutions either:
1. **Wait for ground** - Slow, expensive ground station time, single point of failure
2. **Complex consensus** - Paxos/Raft algorithms that don't handle network partitions well

What if robots could coordinate themselves, like a flock of birds, but with mathematical guarantees?"

### For Business Audiences

"On-orbit servicing is a $4.4 billion market by 2030. Companies like Northrop Grumman are already extending satellite life for $13 million per mission.

But the bottleneck isn't the robots—it's the coordination. Every robot needs constant commands from expensive ground stations. One Starlink constellation pass burns through ground station capacity.

We're building software that lets robot fleets coordinate autonomously. Less ground station time, more robots per mission, operations during blackouts. It's the difference between 2 robots per mission and 20."

---

## The Solution (2 minutes)

### Core Insight

"CRDTs—Conflict-free Replicated Data Types—are data structures that can merge without conflicts. They're used by:
- **Figma** - Real-time design collaboration
- **Apple Notes** - Offline sync
- **Redis** - Distributed databases

We apply this to robotics. Each robot maintains local state. When they can communicate, they merge. The math guarantees convergence—no conflicts, no coordination overhead."

### How It Works

```
Traditional:                    Our Approach:
                               
Robot → Ground → Robot         Robot ←→ Robot
      ↓      ↓                       ↓
   [Waits]  [Waits]              [Working]
   
Commands flow through           Direct peer coordination
central bottleneck              No single point of failure
```

### Technical Differentiator

"Three properties make this work in space:

1. **Partition Tolerance**: Robots keep working during comms blackouts
2. **Eventual Consistency**: State converges when comms resume  
3. **Monotonic Progress**: Work only moves forward, never backwards

These aren't buzzwords—they're mathematical guarantees from the CRDT formalism."

---

## The Proof (2 minutes)

### Simulation Results

"We built a **fair** simulation—the centralized baseline has command buffering, not a strawman:"

| Scenario | CRDT | Centralized | Winner |
|----------|------|-------------|--------|
| LEO (95% reliable, low latency) | 165 steps | 87 steps | Centralized |
| GEO (90% reliable, 3 step latency) | 153 steps | 106 steps | Centralized |
| Lunar (80% reliable, 10 step latency) | 146 steps | 172 steps | **CRDT** |
| Mars (70% reliable, 100 step latency) | 301 steps | 1000+ (timeout) | **CRDT** |

"**Centralized wins when comms are good.** But at Lunar+ distances, it fails. That's our niche."

### Key Chart

*[Show simulation_results.png]*

"The crossover is around Lunar distances. Below that, centralized works. Above that, you need us."

---

## The Market (2 minutes)

### Market Size

| Segment | 2025 | 2030 | Growth |
|---------|------|------|--------|
| On-Orbit Servicing | $1.2B | $4.4B | 30% CAGR |
| Debris Removal | $0.3B | $1.2B | 32% CAGR |
| Space Robotics (total) | $4.5B | $12B | 22% CAGR |

### Target Customers

**Tier 1: Satellite Operators**
- Life extension services
- Refueling
- Module replacement

**Tier 2: Space Agencies**  
- NASA, ESA, JAXA
- Station maintenance
- Lunar/Mars preparation

**Tier 3: Constellation Operators**
- SpaceX, OneWeb, Amazon Kuiper
- Fleet maintenance at scale

### Competitive Landscape

| Company | Focus | Robots | Our Angle |
|---------|-------|--------|-----------|
| Northrop MEV | Life extension | 1 | Multi-robot coordination |
| Astroscale | Debris removal | 1 | Fleet operations |
| Gitai | Manipulation | 1-2 | Swarm coordination |
| ClearSpace | Debris | 1 | Autonomous operations |

"Everyone's building robots. Nobody's solving multi-robot coordination for space."

---

## The Ask (1 minute)

### For Cofounder Conversations

"I'm looking for a technical cofounder with:
- Robotics background (PhD or 5+ years industry)
- Interest in distributed systems
- Appetite for hard problems

What I bring:
- Distributed systems expertise
- Working proof of concept
- Business development drive

The deal:
- Meaningful equity (30-40% negotiable)
- Work evenings/weekends until funded
- Target: Techstars Starburst or NASA SBIR within 6 months

Is this something you'd want to explore?"

### For Advisors

"I'm not asking for investment. I'm asking for:
- 30 minutes of feedback on the technical approach
- Introductions to people who might be interested
- Advice on the space industry landscape

In exchange:
- Advisory equity (0.5-1%) if we formalize
- First look if we raise"

### For Investors (Later)

"We're raising a pre-seed round of $500K to:
- Complete ROS2 integration
- Build hardware partnership
- Run ground-based demo

Use of funds:
- 18 months runway for 2 founders
- Prototype development
- Business development travel

Target milestones:
- NASA SBIR Phase 1 (validates technology)
- 1-2 LOIs from satellite operators
- Techstars Starburst acceptance"

---

## FAQ / Objections

### "Why can't incumbents just add CRDTs?"

"They could. But:
1. Their architectures assume centralized control
2. We have distributed systems DNA—they have hardware DNA
3. First-mover in framing matters for standards/patents
4. We can partner rather than compete"

### "Is the space market real?"

"Northrop paid $7.8B for Orbital ATK. SpaceX is worth $180B. The market is proven. The question is which segment to enter first."

### "Why not start with Earth robotics?"

"We might. Warehouse robotics and drone swarms have the same coordination problem. That's our Plan B. But space has:
- Higher margins
- Fewer competitors
- More acute need (latency is forced, not optional)
- Government funding paths (SBIR)"

### "What about safety?"

"Critical question. Our architecture has:
- Safety supervisor layer (independent of CRDT logic)
- 'Stop-all' emergency broadcast
- Geofencing
- Watchdog timers

Flight software will need formal verification. We're building the concept now, will add rigor with space-experienced cofounder."

### "Why Python?"

"Prototyping speed. Production flight software will be Rust/C++ with formal verification. We're proving the concept, not building flight code."

### "What's your IP strategy?"

"Three layers:
1. Trade secrets (specific implementation)
2. Potential patents (CRDT + robotics application)
3. Speed + relationships (business moat)

We'll formalize IP strategy post-funding with proper counsel."

---

## Conversation Flow

### First Meeting (30 min)

```
0-5 min:  Rapport, background exchange
5-10 min: The 30-second pitch + problem
10-20 min: Demo/simulation results
20-25 min: Their questions
25-30 min: Next steps
```

### Second Meeting (If Interested)

```
- Deep dive on technical approach
- Their concerns/objections
- Role/equity discussion
- Trial project (optional)
```

### Trial Project Ideas

Before committing, potential cofounder could:
- Review CRDT implementation (2-4 hours)
- Add one feature (4-8 hours)
- Write ROS2 integration plan (2-4 hours)

This tests technical fit and working style.

---

## Key Links

- **GitHub**: [link to repo]
- **Demo Video**: [link when created]
- **Blog Post**: [link when published]
- **LinkedIn**: [your profile]
- **Simulation Results**: [link to chart]

---

## Follow-Up Templates

### After Cofounder Meeting

```
Subject: Great chat - CRDT space robotics next steps

Hi [Name],

Thanks for the conversation today. I enjoyed discussing 
[specific thing they said].

As promised, here's:
- GitHub repo: [link]
- Simulation results: [link]
- [Anything else mentioned]

I'm excited about the possibility of working together. 
The next step could be [specific suggestion based on conversation].

Would [day/time] work for a follow-up?

[Your name]
```

### After Advisor Meeting

```
Subject: Thank you - space robotics advice

Hi [Name],

Thank you for taking the time to share your perspective on 
[specific topic]. Your point about [their insight] was 
particularly helpful.

Based on your suggestion, I'm going to [specific action].

If you think of anyone who might be interested in the 
cofounder role, I'd welcome an introduction.

Happy to keep you updated on progress.

[Your name]
```

---

## Metrics To Track

| Metric | Target | Current |
|--------|--------|---------|
| Outreach sent | 50 | 0 |
| Responses received | 10 | 0 |
| Calls scheduled | 5 | 0 |
| Calls completed | 5 | 0 |
| Interested cofounders | 2 | 0 |
| Advisors committed | 2 | 0 |
