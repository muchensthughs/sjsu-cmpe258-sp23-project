import schemdraw
from schemdraw import flow
from enum import Enum
from tools.extract_text import detectText
import pytesseract

## Input: result of detection
##     objects: [(id, x1, y1, x2, y2, classname, confidence)...(id, x1, y1, x2, y2, classname, confidence, head_coord, tail_coord)]
##     arrow2shape: {arrow_id: (head_id, tail_id)}
## Output: generated flow chart image

######
# text -> ignore for now
# arrow -> flow.Arrow()
# data -> flow.Data()
# decision -> flow.Decision
# process -> flow.Box
# terminator -> flow.Start()
# connection -> flow.Circle(r=.5)
#######

def test():
    with schemdraw.Drawing(file='fc_drawing.jpg') as d:
        d += (e := flow.Start().label(''))
        d += flow.Arrow().down(d.unit/2).at(e.S)
        d += (e := flow.Box().label(''))
        d += flow.Arrow().down(d.unit/2).at(e.S)
        d += (decision := flow.Decision().label(''))
        d += flow.Arrow().right(d.unit/2).at(decision.E)
        d += (e := flow.Box().label(''))
        d += flow.Arrow().down(d.unit/2).at(e.S)
        d += (e := flow.Data().label(''))
        d += flow.Arrow().left(d.unit/2).at(e.W)
        d += (e := flow.Start().anchor('E').label(''))
        # d += flow.Arrow().down(d.unit/2).at(decision.S)
        d += flow.Wire('n', k=-1, arrow='->').at(decision.S).to(e.N)
        #  if both shapes are already there, then use wire

def test2():
    with schemdraw.Drawing(file='fc_drawing.jpg') as d:
        d.config(fontsize=11)
        d += (b := flow.Start().label('START'))
        d += flow.Arrow().down(d.unit/2)
        d += (d1 := flow.Decision(w=5, h=3.9, E='YES', S='NO').label('DO YOU\nUNDERSTAND\nFLOW CHARTS?'))
        d += flow.Arrow().length(d.unit/2)
        d += (d2 := flow.Decision(w=5, h=3.9, E='YES', S='NO').label('OKAY,\nYOU SEE THE\nLINE LABELED\n"YES"?'))
        d += flow.Arrow().length(d.unit/2)
        d += (d3 := flow.Decision(w=5.2, h=3.9, E='YES', S='NO').label('BUT YOU\nSEE THE ONES\nLABELED "NO".'))

        d += flow.Arrow().right(d.unit/2).at(d3.E)
        d += flow.Box(w=2, h=1.25).anchor('W').label('WAIT,\nWHAT?')
        d += flow.Arrow().down(d.unit/2).at(d3.S)
        d += (listen := flow.Box(w=2, h=1).label('LISTEN.'))
        d += flow.Arrow().right(d.unit/2).at(listen.E)
        d += (hate := flow.Box(w=2, h=1.25).anchor('W').label('I HATE\nYOU.'))

        d += flow.Arrow().right(d.unit*3.5).at(d1.E)
        d += (good := flow.Box(w=2, h=1).anchor('W').label('GOOD'))
        d += flow.Arrow().right(d.unit*1.5).at(d2.E)
        d += (d4 := flow.Decision(w=5.3, h=4.0, E='YES', S='NO').anchor('W').label('...AND YOU CAN\nSEE THE ONES\nLABELED "NO"?'))

        d += flow.Wire('-|', arrow='->').at(d4.E).to(good.S)
        d += flow.Arrow().down(d.unit/2).at(d4.S)
        d += (d5 := flow.Decision(w=5, h=3.6, E='YES', S='NO').label('BUT YOU\nJUST FOLLOWED\nTHEM TWICE!'))
        d += flow.Arrow().right().at(d5.E)
        d += (question := flow.Box(w=3.5, h=1.75).anchor('W').label("(THAT WASN'T\nA QUESTION.)"))
        d += flow.Wire('n', k=-1, arrow='->').at(d5.S).to(question.S)

        d += flow.Line().at(good.E).tox(question.S)
        d += flow.Arrow().down()
        d += (drink := flow.Box(w=2.5, h=1.5).label("LET'S GO\nDRINK."))
        d += flow.Arrow().right().at(drink.E).label('6 DRINKS')
        d += flow.Box(w=3.7, h=2).anchor('W').label('HEY, I SHOULD\nTRY INSTALLING\nFREEBSD!')
        d += flow.Arrow().up(d.unit*.75).at(question.N)
        d += (screw := flow.Box(w=2.5, h=1).anchor('S').label('SCREW IT.'))
        d += flow.Arrow().at(screw.N).toy(drink.S)


    
class Direction(Enum):
    UP = 'S'
    DOWN = 'N'
    RIGHT = 'W'
    LEFT = 'E'
  
def getDirection(head, tail):
    dx = float(head[0]) - float(tail[0])
    dy = float(head[1])- float(tail[1])
    if dx == 0:
        return Direction.DOWN if dy > 0 else Direction.UP
    gradient = dy / dx
    if dx > 0 and abs(gradient) < 1: return Direction.RIGHT
    if dx < 0 and abs(gradient) < 1: return Direction.LEFT
    if dy >= 0 and abs(gradient) > 1: return Direction.DOWN
    if dy <= 0 and abs(gradient) > 1: return Direction.UP

def drawShape(obj, text, d):
    cls_name = obj[5]
    if cls_name == 'data':
            d += (e := flow.Data().label(text))
    elif cls_name == 'decision':
            d += (e := flow.Decision().label(text))
    elif cls_name == 'process':
            d += (e := flow.Box().label(text))
    elif cls_name == 'terminator':
            d += (e := flow.Start().label(text))
    elif cls_name == 'connection':
            d += (e := flow.Circle(r=.5).label(text))
    return e

def drawArrow(direction, tail, d):
    if direction == Direction.DOWN:
        d += flow.Arrow().down(d.unit/2).at(tail.S)
    elif direction == Direction.UP:
        d += flow.Arrow().up(d.unit/2).at(tail.N)
    elif direction == Direction.RIGHT:
        d += flow.Arrow().right(d.unit/2).at(tail.E)
    elif direction == Direction.LEFT:
        d += flow.Arrow().left(d.unit/2).at(tail.W)
    

def wireArrow(direction, tail, head, d):
    if direction == Direction.DOWN:
        d += flow.Wire('n', k=-1, arrow='->').at(tail.S).to(head.N)
    elif direction == Direction.UP:
        d += flow.Wire('n', k=-1, arrow='->').at(tail.N).to(head.S)
    elif direction == Direction.RIGHT:
        d += flow.Wire('z', k=-1, arrow='->').at(tail.E).to(head.W)
    elif direction == Direction.LEFT:
        d += flow.Wire('z', k=-1, arrow='->').at(tail.W).to(head.E)
        
def identifyTextFromShape(obj_map, text2shape, shapeId, original_image):
    textObj = None
    for textId in text2shape:
        if text2shape[textId] == shapeId: 
            textObj = obj_map[textId]
            break
    return detectText(textObj, original_image)

## with schemdraw we can not specify position, we can only draw graph based on the flow relationships
## Assume shapes are all connected by arrows...
## we need to associate shapes in the graph with the object id
def draw_from_detection(objects, arrow2shape, text2shape, original_image):
    # convert objects into map 
    obj_map = {}
    for obj in objects:
        obj_map[obj[0]] = obj
    shape_map = {}
    
    arrow_queue = []
    for arrow_id in arrow2shape:
         arrow_queue.append((arrow_id, arrow2shape[arrow_id][0], arrow2shape[arrow_id][1])) # (arrow_id, head_id, tail_id)
    # for each arrow, draw it's connecting tail shape, then arrow, then head shape
    with schemdraw.Drawing(file='fc_drawing.jpg') as d:
        while len(arrow_queue) > 0:
            # print(arrow_queue)
            arrow = arrow_queue[0]
            arrow_id = arrow[0]
            head_id = arrow[1] 
            tail_id = arrow[2]

            # get arrow direction
            head_coord = obj_map[arrow_id][7]
            tail_coord = obj_map[arrow_id][8]
            direction = getDirection(head_coord, tail_coord)

            # if both shapes are already there, wire them
            if (tail_id in shape_map and head_id in shape_map):
                wireArrow(direction, shape_map[tail_id], shape_map[head_id], d) 
                arrow_queue.pop(0)                  
            elif (tail_id in shape_map and head_id not in shape_map):
                tail = shape_map[tail_id]
                drawArrow(direction, tail, d)
                text = identifyTextFromShape(obj_map, text2shape, head_id, original_image)
                head = drawShape(obj_map[head_id], text, d)
                shape_map[head_id] = head
                arrow_queue.pop(0) 
            elif (tail_id not in shape_map and head_id in shape_map):
                # skip this one for now
                arrow_queue.pop(0) 
                arrow_queue.append(arrow)
            else:
                if (obj_map[tail_id][5] == 'terminator' or obj_map[tail_id][5] == 'connection'): 
                    text = identifyTextFromShape(obj_map, text2shape, tail_id, original_image)
                    tail = drawShape(obj_map[tail_id], text, d)
                    shape_map[tail_id] = tail 
                    drawArrow(direction, tail, d)
                    text = identifyTextFromShape(obj_map, text2shape, head_id, original_image)
                    head = drawShape(obj_map[head_id], text, d)
                    shape_map[head_id] = head
                    arrow_queue.pop(0)
                else:
                    arrow_queue.pop(0) 
                    arrow_queue.append(arrow) 









