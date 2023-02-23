{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use record patterns" #-}
{-# LANGUAGE LambdaCase #-}

module Data.Accelerate.TensorFlow.Operation where

import Data.Array.Accelerate.AST.Operation
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Backend
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Trafo.Desugar
import Data.Array.Accelerate.Trafo.Exp.Substitution
import Data.Type.Equality
import Data.Array.Accelerate.Lifetime
import Foreign
import Data.Array.Accelerate.AST.Kernel (NoKernelMetadata)
import Data.Text.Prettyprint.Doc
import Data.Array.Accelerate.Pretty.Exp
    ( prettyConst, primOperator )
import Data.Array.Accelerate.Pretty.Print (Operator(..))
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Smart (typeR, undef)
import GHC.Conc (TVar(TVar))
import Data.Array.Accelerate.Pretty.Print (Adoc)

data TensorOp op where
  TConstant :: ScalarType s -> s -> TensorOp (Out sh s -> ())
  TPrimFun :: PrimFun (a -> b) -> TensorOp (In sh a -> Out sh b -> ())
  TId :: ScalarType s -> TensorOp (In sh s -> Out sh s -> ())
  TWhere :: TensorOp (In sh a -> Out DIM1 sh -> ())
  TTensorScatter :: ScatterFun -> TensorOp (Mut sh' s -> In sh sh' -> In sh s -> ())
  TBooleanMask :: ScalarType s -> TensorOp (In DIM1 s -> In DIM1 PrimBool -> Out DIM1 s -> ())

instance PrettyOp TensorOp where
  prettyOp (TConstant s e)    = vsep ["TConst", prettyConst (TupRsingle s) e]
  prettyOp (TPrimFun f)       = vsep ["TBinOp", opName (primOperator f) ]
  prettyOp (TId s)            = vsep ["TId", prettyScalarType  s]
  prettyOp TWhere             = "TWhere"
  prettyOp (TTensorScatter f) = vsep ["TTensorScatter", viaShow f]
  prettyOp (TBooleanMask s)   = vsep ["TBooleanMask", prettyScalarType s]

instance NFData' TensorOp where
  rnf' !_ = ()

instance DesugarAcc TensorOp where
  mkMap (ArgFun (Lam lhs (Body body))) (ArgArray _ (ArrayR _ t) _ gvb) aOut =
    mkMapF (push' Empty (lhs, distributeBIdx t gvb)) body aOut
  mkMap _ _ _ = error "impossible"

  mkGenerate f aOut@(ArgArray _ (ArrayR sh _) gv gvb)
    | DeclareVars lhs w k <- declareVars $ buffersR (TupRsingle scalarTypeInt)
    , DeclareVars lhs' w' k' <- declareVars $ buffersR (shapeType sh)
    = -- 1) Creating the shape array
      -- 1.1) Create a Tensor of shape sh with only ones
      aletUnique lhs (desugarAlloc (ArrayR sh (TupRsingle scalarTypeInt)) (fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        (TConstant scalarTypeInt 1)
        (ArgArray Out (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars w gv) (k weakenId) :>: ArgsNil)
      ) $
      -- 1.2) Apply tf.where to obtain a 1D array of indexes
      aletUnique lhs' (desugarAlloc (ArrayR dim1 (shapeType sh)) (toDIM1 gv)) $ -- wat te doen met sh ~ dim1 voor gv?
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        TWhere
        (ArgArray In (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars (w' .> w) gv) (k w') :>:
         ArgArray Out (ArrayR dim1 (shapeType sh)) (weakenVars (w' .> w) gv) (k' weakenId) :>: -- wat te doen met sh ~ dim1 voor gv?
         ArgsNil)
      ) $
      _ -- en hoe nu verder?
      
       -- TODO: filling with indices
      --mkMap (weaken w f) (ArgArray In (ArrayR sh sh') (weakenVars w gv) (k weakenId)) (weaken w aOut)

  mkPermute -- TODO: support all shapes (not only dim1)
    (ArgFun comb)
    defaults@(ArgArray _ (ArrayR (ShapeRsnoc ShapeRz) _) gv' gvb')
    perm
    source@(ArgArray _ (ArrayR (ShapeRsnoc ShapeRz) t) gv gvb) -- reshape x compute (shapeSize)
    | sh <- ShapeRsnoc ShapeRz
    , maybeSh <- TupRpair (TupRsingle scalarTypeWord8) (TupRpair TupRunit (shapeType sh))
    , DeclareVars lhs w k  <- declareVars $ buffersR maybeSh
    , DeclareVars lhs' w' k' <- declareVars $ buffersR (shapeType sh)
    , DeclareVars lhs'' w'' k'' <- declareVars $ buffersR t
    = -- 1) allocate permute indices array n
      aletUnique lhs (desugarAlloc (ArrayR sh maybeSh) (fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkGenerate
        (weaken w perm)
        (ArgArray Out (ArrayR sh maybeSh) (weakenVars w gv) (k weakenId))
      ) $
      aletUnique lhs' (desugarAlloc (ArrayR sh (shapeType sh)) (fromGrounds (weakenVars w gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit -- toIndex
      (booleanMask (shapeType sh)                                                  
       (ArgArray In (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars (w' .> w) gv) (isJust (TupRpair TupRunit (shapeType sh)) (k w')))
       (fromJust (shapeType sh) (k w'))
       (k' weakenId)
      ) $
      aletUnique lhs'' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask t
       (ArgArray In (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars (w'' .> w' .> w) gv) (isJust (TupRpair TupRunit (shapeType sh)) (k (w'' .> w'))))
       (weakenVars (w'' .> w' .> w) gvb)
       (k'' weakenId)
      ) $
      Exec (TTensorScatter scatterFun) (
        ArgArray Mut (ArrayR dim1 t) (weakenVars (w'' .> w' .> w) gv') (weakenVars (w'' .> w' .> w) gvb') :>: 
        ArgArray In  (ArrayR dim1 (shapeType sh)) (weakenVars (w'' .> w' .> w) gv) (k' w'') :>:
        ArgArray In  (ArrayR dim1 t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId) :>: 
        ArgsNil)
        where scatterFun = case comb of
                Lam (LeftHandSideSingle _) (Lam (LeftHandSideSingle _) (Body (PrimApp (PrimAdd _) (Pair (Evar (Var _ ZeroIdx)) (Evar (Var _ (SuccIdx ZeroIdx))))))) -> ScatterFunAdd -- swap idx?
                _ -> error "complex combination for permute not supported" 
  mkPermute _ _ _ _ = undefined

instance SimplifyOperation TensorOp where

instance SLVOperation TensorOp where
  slvOperation _ = Nothing

instance ShrinkArg (BackendClusterArg TensorOp) where
  shrinkArg = undefined -- TODO: David
  deadArg = undefined -- TODO: David

instance NFData' (BackendClusterArg TensorOp) where
  rnf' = undefined -- TODO: David

-- instance MakesILP TensorOp where TODO: David

booleanMask :: TypeR a -> Arg env (In DIM1 Word8) -> GroundVars env (Buffers a) -> GroundVars env (Buffers a) -> PreOpenAcc TensorOp env ()
booleanMask TupRunit _ _ gvb = Return gvb
booleanMask t@(TupRsingle s) aOut@(ArgArray _ _ gv _) gvbIn gvbOut = Exec (TBooleanMask s) (ArgArray In (ArrayR dim1 t) gv gvbIn :>: aOut :>: ArgArray Out (ArrayR dim1 t) gv gvbOut :>: ArgsNil)
booleanMask (TupRpair t1 t2) aOut (TupRpair gvbIn1 gvbIn2) (TupRpair gvbOut1 gvbOut2) = Alet (LeftHandSideWildcard TupRunit) TupRunit 
 (booleanMask t1 aOut gvbIn1 gvbOut1)
 (booleanMask t2 aOut gvbIn2 gvbOut2)
booleanMask _ _ _ _ = error "impossible"

isJust :: TypeR a -> GroundVars env (Buffers (Word8, a)) -> GroundVars env (Buffers Word8)
isJust _ (TupRpair word8 _) = word8
isJust _ _ = error "impossible"

fromJust :: TypeR a -> GroundVars env (Buffers (Word8, ((), a))) -> GroundVars env (Buffers a)
fromJust _ (TupRpair _ (TupRpair _ a)) = a
fromJust _ _ = error "impossible"

mkMapF :: forall env env' sh t. BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' t
  -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapF _ (Const s e) aOut = Exec (TConstant s e) $ aOut :>: ArgsNil

mkMapF env (PrimApp f exp) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF -- flatten higher-order expression
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   Exec -- apply method to the result
    (TPrimFun f)
    (
      ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
      weaken w aOut :>:
      ArgsNil
    )

mkMapF env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp1
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF
     (weakenEnv w env)
     (weakenArrayInstr w exp1)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   mkMapF
     (push' (weakenEnv w env) (elhs, distributeBIdx a (k weakenId)))
     (weakenArrayInstr w exp2)
     (weaken w aOut)

mkMapF env (Evar (Var s idx)) (ArgArray _ arrayR gv gvb@(TupRsingle (Var groundR _)))
  | Refl <- reprIsSingle @ScalarType @t @Buffer s
  = let (BIdx idx') = prj' idx env
        gvb'        = TupRsingle (Var groundR idx')
    in  Exec
          (TId s)
          (
            ArgArray In arrayR gv gvb' :>:
            ArgArray Out arrayR gv gvb :>:
            ArgsNil
          )

mkMapF env (Pair exp1 exp2) (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2))
 = Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF env exp1 (ArgArray Out (ArrayR sh t1) gv gvb1))
   (mkMapF env exp2 (ArgArray Out (ArrayR sh t2) gv gvb2))

-- TODO

mkMapF _ (Foreign _ _ _ _) _ = undefined
mkMapF _ Nil _ = Return TupRunit
--mkMapF env Nil (ArgArray _ arrayR gv gvb) = Return gvb
-- TId (ArgArray In arrayR gv gvb :>: ArgArray Out arrayR gv gvb :>: ArgsNil)
--mkMapF _ Nil (ArgArray _ _ _ gvb) = Return gvb
mkMapF _ (VecPack _ _) _ = undefined
mkMapF _ (VecUnpack _ _) _ = undefined
mkMapF _ (IndexSlice _ _ _) _ = undefined
mkMapF _ (IndexFull _ _ _) _ = undefined
mkMapF _ (ToIndex _ _ _) _ = undefined -- hoe bewijs ik dat sh = sh?
mkMapF _ (FromIndex _ _ _) _ = undefined
mkMapF _ (Case _ _ _) _ = undefined
mkMapF _ (Cond _ _ _) _ = undefined
mkMapF _ (While _ _ _) _ = undefined
mkMapF _ (PrimConst _) _ = undefined
mkMapF _ (ArrayInstr _ _) _ = undefined
mkMapF _ (ShapeSize _ _) _ = undefined
mkMapF _ (Undef _) _ = undefined
mkMapF _ (Coerce _ _ _) _ = undefined
mkMapF _ _ _ = error "impossible"


-- temp kernel for testing purposes

newtype BufferIdx benv a = BIdx (Idx benv (Buffer a))

instance Sink BufferIdx where
  weaken w (BIdx idx) = BIdx (weaken w idx)

-- benv is buffer env
-- env is scalar env
-- in env staat een int, en en benv kan ik op zoek naar een buffer van ints
type BufferEnv benv env = Env (BufferIdx benv) env

weakenEnv :: Sink f => (env1 :> env') -> Env (f env1) env2 -> Env (f env') env2
weakenEnv w = mapEnv (weaken w)

-- forall is alleen nodig als je @s wilt gebruiken in de method definition
distributeBIdx :: forall env s. TypeR s -> GroundVars env (Buffers s) -> Distribute (BufferIdx env) s
distributeBIdx TupRunit _ = ()
distributeBIdx (TupRsingle s) (TupRsingle (Var _ idx))
  | Refl <- reprIsSingle @ScalarType @s @(BufferIdx env) s
  , Refl <- reprIsSingle @ScalarType @s @Buffer s
  = BIdx idx
distributeBIdx (TupRpair l1 l2) (TupRpair r1 r2) = (distributeBIdx l1 r1, distributeBIdx l2 r2)
distributeBIdx _ _ = error "impossible"

data ScatterFun where
  ScatterFunAdd :: ScatterFun
  ScatterFunMin :: ScatterFun
  deriving Show

prettyScatterFun :: ScatterFun -> Adoc
prettyScatterFun ScatterFunAdd = "ScatterFunAdd"
prettyScatterFun ScatterFunMin = "ScatterFunMin"

toDIM1 :: GroundVars env sh -> ExpVars env' DIM1
toDIM1 = _